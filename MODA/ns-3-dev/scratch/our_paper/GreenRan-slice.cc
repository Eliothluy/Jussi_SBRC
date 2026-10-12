/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * GreenRAN Slice-Aware 5G/NR Simulation — ns-3 / 5G-LENA
 *
 * Slice-aware scheduler with POSIX semaphore IPC for DRL agent control
 * via file exchange (rslaq-kpms.txt, rslaq_actions_for_ns3.csv).
 *
 * Topology: 1 gNB, N UEs (VIDEO_EMBB + SENSOR_MMTC)
 * VIDEO_EMBB is modeled as uplink-heavy video streaming (cameras to server)
 * with light downlink feedback. SENSOR_MMTC is modeled as small periodic sensor reports.
 *
 * Model scope: this is RAN-side / MAC-level slicing by UE/RNTI in the NR
 * scheduler. It approximates end-to-end service separation through application
 * traffic classes, EPC bearers/QoS flows, and RAN-side MAC slicing. It does not
 * model a complete 5GC with NSSF/AMF/SMF/UPF slicing or SDAP.
 */

#include "mac-scheduler.h"

#include "ns3/antenna-module.h"
#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/nr-module.h"
#include "ns3/point-to-point-module.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <cerrno>
#include <exception>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <semaphore.h>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("GreenRanSim");

static const uint32_t NUM_SLICES = 3;
static const uint32_t VIDEO_EMBB_SLICE = 0;
static const uint32_t SENSOR_MMTC_SLICE = 1;
static const uint32_t GENERIC_EMBB_SLICE = 2;

struct GreenRanScenarioConfig
{
    std::string name;
    uint32_t videoUes;
    uint32_t sensorUes;
    uint32_t genericUes;
    double videoUlRateMbps;
    uint32_t videoPktSize;
    uint32_t sensorPktSize;
    double sensorIntervalSec;
    double videoDlFeedbackRateKbps;
    double genericDlRateMbps;
    std::vector<double> weights;
};

struct ProjectSliceProfile
{
    uint32_t sliceId;
    std::string sliceName;
    std::string applicationName;
    std::string projectUseCase;
    std::string serviceType;
    std::string primaryDirection;
    NrQosFlow::FiveQi fiveQi;
    double dlThroughputSlaMbps;
    double ulThroughputSlaMbps;
    double delaySlaMs;
    double pdrSla;
    std::string qosNotes;
};

struct RadioUnitProfile
{
    std::string name;
    double minFrequencyHz;
    double maxFrequencyHz;
    double defaultCenterFrequencyHz;
    double maxBandwidthHz;
    uint16_t numerology;
    double scsKHz;
    uint32_t txPorts;
    uint32_t rxPorts;
    double txPowerPerPortDbm;
    double aggregateTxPowerDbm;
    double receiverNoiseFigureDb;
    double typicalPowerConsumptionW;
    std::string ipRating;
    std::string deployment;
    std::string oRanSplit;
    std::string notes;
};

struct AntennaProfile
{
    std::string name;
    double minFrequencyHz;
    double maxFrequencyHz;
    uint32_t ports;
    double gainDbi;
    double maxGainDbi;
    double azimuthBeamwidthDeg;
    double elevationBeamwidthDeg;
    double electricalDowntiltDeg;
    double minElectricalDowntiltDeg;
    double maxElectricalDowntiltDeg;
    double frontToBackRatioDb;
    double crossPolarDiscriminationDb;
    double isolationDb;
    std::string polarization;
    std::string antennaType;
    std::string notes;
};

static std::map<std::string, GreenRanScenarioConfig>
InitGreenRanScenarios()
{
    std::map<std::string, GreenRanScenarioConfig> m;
    // Sensor intervals reduced to ensure mMTC traffic is visible to the
    // scheduler within a 10 s simulation (each sensor sends 3-10 packets).
    // Previous intervals (5-60 s) produced 0-1 packets per sensor — the
    // scheduler never saw SENSOR_MMTC UEs as active.
    m["greenran_low"] = {"greenran_low", 5, 8, 5, 10.0, 1200, 80, 2.0, 64.0, 5.0, {0.5, 0.2, 0.3}};
    m["greenran_normal"] = {"greenran_normal", 5, 8, 20, 15.0, 1200, 80, 2.0, 128.0, 5.0, {0.4, 0.15, 0.45}};
    m["greenran_video_heavy"] = {"greenran_video_heavy", 5, 8, 15, 25.0, 1400, 80, 2.0, 128.0, 8.0, {0.5, 0.1, 0.4}};
    m["greenran_congestion"] = {"greenran_congestion", 5, 8, 30, 25.0, 1400, 100, 1.0, 128.0, 2.0, {0.35, 0.1, 0.55}};
    m["greenran_night_energy"] = {"greenran_night_energy", 5, 8, 3, 10.0, 1200, 80, 3.0, 64.0, 3.0, {0.4, 0.3, 0.3}};
    m["greenran_balanced"] = {"greenran_balanced", 5, 8, 25, 15.0, 1200, 80, 2.0, 64.0, 5.0, {0.35, 0.15, 0.5}};
    return m;
}

static RadioUnitProfile
MakeRan650N78Profile()
{
    return {"Benetel RAN650 n78",
            3.3e9,
            3.8e9,
            3.55e9,
            100e6,
            1,
            30.0,
            4,
            4,
            37.0,
            37.0 + 10.0 * std::log10(4.0),
            4.0,
            100.0,
            "IP65",
            "Outdoor",
            "7.2x",
            "Outdoor O-RU, n78, 4T4R, 4 N-type RF ports, 2x10GbE SFP+ fronthaul, GPS/PTP sync, -48V DC."};
}

static AntennaProfile
MakeAlphaAw3161Profile(double downtiltDeg)
{
    return {"Alpha Wireless AW3161-E-F-V2",
            3.3e9,
            3.8e9,
            4,
            17.4,
            17.9,
            65.0,
            7.0,
            downtiltDeg,
            0.0,
            10.0,
            30.0,
            18.0,
            28.0,
            "+/-45 slant linear",
            "Outdoor sector panel",
            "4-port 3.5 GHz panel with eRET; exact measured radiation pattern is not embedded."};
}

static RadioUnitProfile
MakeLiteOnFlexFiIndoorProfile()
{
    return {"LiteOn FlexFi Sub6 O-RU",
            3.3e9,
            3.8e9,
            3.55e9,
            100e6,
            1,
            30.0,
            4,
            4,
            24.0,
            24.0 + 10.0 * std::log10(4.0),
            0.0,
            65.0,
            "IP20",
            "Indoor",
            "7.2x",
            "Indoor small-cell O-RU with internal antenna, 1 W total class, -5C to 45C operating range."};
}

static std::vector<ProjectSliceProfile>
MakeUfpaGreenRanSliceProfiles(double videoRateMbps, double videoFeedbackKbps, uint32_t videoUes,
                               double genericRateMbps, uint32_t genericUes)
{
    std::vector<ProjectSliceProfile> profiles(NUM_SLICES);
    profiles[VIDEO_EMBB_SLICE] = {VIDEO_EMBB_SLICE,
                                  "VIDEO_EMBB",
                                  "App1-Vigilancia",
                                  "Campus UFPA Belem: video 4K/H.265 e IA para seguranca publica",
                                  "eMBB low-latency video",
                                  "UL",
                                  NrQosFlow::NGBR_LOW_LAT_EMBB,
                                  0.95 * videoFeedbackKbps * videoUes / 1000.0,
                                  25.0,
                                  100.0,
                                  0.99,
                                  "Aproxima 5QI 80 para eMBB de baixa latencia; cameras 4K requerem cerca de 25 Mbps e <100 ms."};
    profiles[SENSOR_MMTC_SLICE] = {SENSOR_MMTC_SLICE,
                                   "SENSOR_MMTC",
                                   "App2-Monitoramento",
                                   "Agro/Campus: sensores ambientais e de solo com coleta intermitente",
                                   "mMTC sensor monitoring",
                                   "UL",
                                   NrQosFlow::NGBR_MC_DATA,
                                   0.0,
                                   0.0,
                                   5000.0,
                                   0.95,
                                   "Aproxima 5QI 70 para dados criticos/monitoramento; foco em PDR, escala e uso eficiente de PRBs."};
    profiles[GENERIC_EMBB_SLICE] = {GENERIC_EMBB_SLICE,
                                    "GENERIC_EMBB",
                                    "App3-Usuarios",
                                    "Campus UFPA: usuarios comuns com streaming e VoIP",
                                    "eMBB mixed traffic",
                                    "DL+UL",
                                    NrQosFlow::NGBR_VIDEO_TCP_PREMIUM,
                                    0.9 * genericRateMbps * genericUes,
                                    0.064 * genericUes,
                                    150.0,
                                    0.98,
                                    "Aproxima 5QI 4 para streaming e 5QI 1 para VoIP; trafego misto de usuarios de campus."};
    return profiles;
}

static uint32_t
FiveQiToNumber(NrQosFlow::FiveQi fiveQi)
{
    return static_cast<uint32_t>(static_cast<uint8_t>(fiveQi));
}

static Ptr<NrQosRule>
MakeUeAddressScopedQosRule(const Ipv4Address& remoteHostAddr,
                           const Ipv4Address& ueAddr,
                           uint8_t precedence)
{
    Ptr<NrQosRule> rule = Create<NrQosRule>();
    NrQosRule::PacketFilter filter;
    filter.direction = NrQosRule::BIDIRECTIONAL;
    filter.remoteAddress = remoteHostAddr;
    filter.remoteMask = Ipv4Mask("255.255.255.255");
    filter.localAddress = ueAddr;
    filter.localMask = Ipv4Mask("255.255.255.255");
    filter.remotePortStart = 0;
    filter.remotePortEnd = 65535;
    filter.localPortStart = 0;
    filter.localPortEnd = 65535;
    filter.typeOfService = 0;
    filter.typeOfServiceMask = 0;
    rule->SetPrecedence(precedence);
    rule->Add(filter);
    return rule;
}

static std::vector<double>
ParseWeights(const std::string& str)
{
    std::vector<double> w;
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, ','))
    {
        w.push_back(std::stod(token));
    }
    return w;
}

static std::string
WeightsToString(const std::vector<double>& weights)
{
    std::ostringstream oss;
    for (uint32_t i = 0; i < weights.size(); ++i)
    {
        if (i > 0)
        {
            oss << ",";
        }
        oss << std::fixed << std::setprecision(4) << weights[i];
    }
    return oss.str();
}

static bool
CliArgProvided(int argc, char* argv[], const std::string& name)
{
    const std::string prefix = "--" + name;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg(argv[i]);
        if (arg == prefix || arg.find(prefix + "=") == 0)
        {
            return true;
        }
    }
    return false;
}

static void
ValidateWeights(const std::vector<double>& weights)
{
    NS_ABORT_MSG_IF(weights.size() != NUM_SLICES,
                    "GreenRAN requires exactly " << NUM_SLICES << " weights, got " << weights.size());
    double sum = 0.0;
    for (double w : weights)
    {
        NS_ABORT_MSG_IF(w < 0.0, "Slice weights must be non-negative");
        sum += w;
    }
    NS_ABORT_MSG_IF(std::abs(sum - 1.0) > 1e-6,
                    "Slice weights must sum to 1.0, got " << sum);
}

static void
EnsureDirectoryExists(const std::string& path)
{
    if (path.empty() || path == ".")
    {
        return;
    }

    struct stat st;
    if (stat(path.c_str(), &st) == 0)
    {
        NS_ABORT_MSG_IF(!S_ISDIR(st.st_mode), "Output path exists but is not a directory: " << path);
        return;
    }

    int ret = mkdir(path.c_str(), 0775);
    NS_ABORT_MSG_IF(ret != 0 && errno != EEXIST,
                    "Failed to create output directory '" << path << "' errno=" << errno);
}

static double
ComputeThermalNoiseDbm(double bandwidthHz, double noiseFigureDb)
{
    return -174.0 + 10.0 * std::log10(bandwidthHz) + noiseFigureDb;
}

static bool
IsOutdoorHardwareProfile(const std::string& hardwareProfile)
{
    return hardwareProfile == "ran650_aw3161_outdoor";
}

static bool
IsIndoorHardwareProfile(const std::string& hardwareProfile)
{
    return hardwareProfile == "liteon_flexfi_indoor";
}

static void
ValidateDatasheetConfig(const RadioUnitProfile& radio,
                        const AntennaProfile& antenna,
                        const std::string& hardwareProfile,
                        const std::string& deploymentScenario,
                        double centralFrequency,
                        double bandwidth,
                        uint16_t numerology,
                        bool enableHardwareValidation,
                        bool allowOutOfDatasheetConfig)
{
    if (!enableHardwareValidation)
    {
        std::cout << "WARNING: hardware datasheet validation disabled by CLI.\n";
        return;
    }

    auto handleViolation = [allowOutOfDatasheetConfig](const std::string& message)
    {
        if (allowOutOfDatasheetConfig)
        {
            std::cout << "WARNING: out-of-datasheet hardware configuration: " << message << "\n";
        }
        else
        {
            NS_ABORT_MSG(message);
        }
    };

    if (centralFrequency < radio.minFrequencyHz || centralFrequency > radio.maxFrequencyHz ||
        centralFrequency < antenna.minFrequencyHz || centralFrequency > antenna.maxFrequencyHz)
    {
        handleViolation("RAN650/AW3161 profile requires n78 frequency range: 3.3-3.8 GHz");
    }
    if (bandwidth > radio.maxBandwidthHz)
    {
        handleViolation("RAN650 supports up to 100 MHz instantaneous bandwidth");
    }
    if (numerology != radio.numerology)
    {
        handleViolation("RAN650 datasheet specifies SCS 30 kHz, which corresponds to numerology 1");
    }
    if (IsOutdoorHardwareProfile(hardwareProfile) && deploymentScenario == "indoor")
    {
        handleViolation("Outdoor RAN650/AW3161 profile should not be paired with deploymentScenario=indoor");
    }
    if (IsIndoorHardwareProfile(hardwareProfile) && deploymentScenario != "indoor")
    {
        handleViolation("LiteOn FlexFi profile is indoor/IP20 and should not be used for outdoor results");
    }
}

static std::string
SliceName(uint32_t idx)
{
    if (idx == VIDEO_EMBB_SLICE) return "VIDEO_EMBB";
    if (idx == SENSOR_MMTC_SLICE) return "SENSOR_MMTC";
    if (idx == GENERIC_EMBB_SLICE) return "GENERIC_EMBB";
    return "UNKNOWN";
}

enum class Direction
{
    UL,
    DL,
    Unknown
};

static std::string
DirectionToString(Direction direction)
{
    if (direction == Direction::UL)
    {
        return "UL";
    }
    if (direction == Direction::DL)
    {
        return "DL";
    }
    return "?";
}

static Direction
ClassifyDirection(const Ipv4FlowClassifier::FiveTuple& tuple,
                  const Ipv4Address& remoteHostAddr)
{
    if (tuple.sourceAddress == remoteHostAddr)
    {
        return Direction::DL;
    }
    if (tuple.destinationAddress == remoteHostAddr)
    {
        return Direction::UL;
    }
    return Direction::Unknown;
}

struct FlowRecord
{
    uint16_t ueId = 0;
    uint32_t sliceId = 0;
    Direction direction = Direction::Unknown;
    uint64_t txBytes = 0;
    uint64_t rxBytes = 0;
    uint32_t txPackets = 0;
    uint32_t rxPackets = 0;
    uint32_t lostPackets = 0;
    double delaySumSec = 0.0;
};

static std::vector<FlowRecord>
ExtractFlowRecords(const FlowMonitor::FlowStatsContainer& stats,
                   const Ptr<Ipv4FlowClassifier>& classifier,
                   const std::map<uint16_t, uint16_t>& portToUeId,
                   const std::map<uint16_t, uint32_t>& ueToSliceIdx,
                   const Ipv4Address& remoteHostAddr)
{
    std::vector<FlowRecord> records;
    for (const auto& kv : stats)
    {
        Ipv4FlowClassifier::FiveTuple tuple = classifier->FindFlow(kv.first);
        if (tuple.protocol != 17)
        {
            continue;
        }

        auto itPort = portToUeId.find(tuple.destinationPort);
        if (itPort == portToUeId.end())
        {
            continue;
        }

        const uint16_t ueId = itPort->second;
        auto itSlice = ueToSliceIdx.find(ueId);
        if (itSlice == ueToSliceIdx.end())
        {
            continue;
        }

        const Direction direction = ClassifyDirection(tuple, remoteHostAddr);
        if (direction == Direction::Unknown)
        {
            continue;
        }

        const FlowMonitor::FlowStats& st = kv.second;
        FlowRecord record;
        record.ueId = ueId;
        record.sliceId = itSlice->second;
        record.direction = direction;
        record.txBytes = st.txBytes;
        record.rxBytes = st.rxBytes;
        record.txPackets = st.txPackets;
        record.rxPackets = st.rxPackets;
        record.lostPackets = st.lostPackets;
        record.delaySumSec = st.delaySum.GetSeconds();
        records.push_back(record);
    }
    return records;
}

struct DeploymentConfig
{
    std::string requestedName;
    std::string channelScenario;
    double defaultRadiusM = 50.0;
    double defaultGnbHeightM = 10.0;
    double defaultUeHeightM = 1.5;
    bool defaultShadowing = false;
};

static DeploymentConfig
GetDeploymentConfig(const std::string& deploymentScenario)
{
    if (deploymentScenario == "umi" || deploymentScenario == "UMi" ||
        deploymentScenario == "indoor_small" || deploymentScenario == "campus_short")
    {
        return {deploymentScenario, "UMi", 80.0, 10.0, 1.5, true};
    }
    if (deploymentScenario == "uma" || deploymentScenario == "UMa" ||
        deploymentScenario == "outdoor_medium" || deploymentScenario == "fwa")
    {
        return {deploymentScenario, "UMa", 250.0, 25.0, 1.5, true};
    }
    if (deploymentScenario == "indoor" || deploymentScenario == "IndoorOffice" ||
        deploymentScenario == "InH-OfficeOpen")
    {
        return {deploymentScenario, "InH-OfficeOpen", 30.0, 3.0, 1.5, false};
    }
    NS_FATAL_ERROR("Unknown deploymentScenario: " << deploymentScenario
                   << ". Use umi, uma, indoor, indoor_small, campus_short, outdoor_medium, or fwa.");
}

static uint32_t
EstimateNrPrbs(double bandwidthHz, uint16_t numerology)
{
    const double scsHz = 15000.0 * std::pow(2.0, numerology);
    const double rbHz = 12.0 * scsHz;
    return static_cast<uint32_t>(std::floor((bandwidthHz * 0.982) / rbHz));
}

static uint32_t
EstimateRbsPerRbg(uint32_t prbs)
{
    if (prbs <= 36)
    {
        return 2;
    }
    if (prbs <= 72)
    {
        return 4;
    }
    if (prbs <= 144)
    {
        return 8;
    }
    return 16;
}

struct SliceSla
{
    double dlThroughputMbps = 0.0;
    double ulThroughputMbps = 0.0;
    double delayMs = 0.0;
    double pdr = 0.0;
};

struct DrlSliceAction
{
    uint64_t timestampMs = 0;
    std::vector<double> weights;
};

static uint64_t g_lastAppliedActionTimestamp = 0;

static bool
TryParseDrlActionFile(const std::string& path, DrlSliceAction& action)
{
    std::ifstream in(path, std::ios::in);
    if (!in.is_open())
    {
        return false;
    }

    std::string line;
    std::string lastDataLine;
    while (std::getline(in, line))
    {
        if (!line.empty() && line.find_first_not_of(" \t\r\n") != std::string::npos &&
            line.find("timestamp") == std::string::npos)
        {
            lastDataLine = line;
        }
    }
    if (lastDataLine.empty())
    {
        return false;
    }

    std::vector<std::string> cols;
    std::stringstream ss(lastDataLine);
    std::string token;
    while (std::getline(ss, token, ','))
    {
        cols.push_back(token);
    }
    if (cols.size() < 3)
    {
        return false;
    }

    try
    {
        uint64_t timestamp = static_cast<uint64_t>(std::stoull(cols[0]));
        if (timestamp <= g_lastAppliedActionTimestamp)
        {
            return false;
        }

        std::vector<double> weights(NUM_SLICES, 0.0);
        bool perSliceRows = false;
        for (uint32_t i = 0; i < NUM_SLICES && cols.size() >= 3; ++i)
        {
            // Supported row format from ns-o-ran-gym:
            // timestamp,sliceId,dedicatedPRB,minPRB,maxPRB
            if (cols.size() >= 3)
            {
                uint32_t sliceId = static_cast<uint32_t>(std::stoul(cols[1]));
                if (sliceId < NUM_SLICES)
                {
                    double dedicatedPrbPct = std::stod(cols[2]);
                    weights[sliceId] = dedicatedPrbPct / 100.0;
                    perSliceRows = true;
                    break;
                }
            }
        }

        if (perSliceRows)
        {
            // Re-read all rows for the same timestamp to collect each slice.
            in.clear();
            in.seekg(0, std::ios::beg);
            while (std::getline(in, line))
            {
                if (line.find("timestamp") != std::string::npos)
                {
                    continue;
                }
                std::vector<std::string> row;
                std::stringstream rowSs(line);
                while (std::getline(rowSs, token, ','))
                {
                    row.push_back(token);
                }
                if (row.size() < 3)
                {
                    continue;
                }
                uint64_t rowTs = static_cast<uint64_t>(std::stoull(row[0]));
                if (rowTs != timestamp)
                {
                    continue;
                }
                uint32_t sliceId = static_cast<uint32_t>(std::stoul(row[1]));
                if (sliceId < NUM_SLICES)
                {
                    weights[sliceId] = std::stod(row[2]) / 100.0;
                }
            }
        }
        else if (cols.size() >= NUM_SLICES + 1)
        {
            for (uint32_t i = 0; i < NUM_SLICES; ++i)
            {
                weights[i] = std::stod(cols[i + 1]);
            }
        }

        double sum = 0.0;
        for (double w : weights)
        {
            if (w < 0.0)
            {
                return false;
            }
            sum += w;
        }
        if (sum <= 0.0)
        {
            return false;
        }
        for (double& w : weights)
        {
            w /= sum;
        }

        action.timestampMs = timestamp;
        action.weights = weights;
        return true;
    }
    catch (const std::exception&)
    {
        return false;
    }
}

// ---------------------------------------------------------------------------
// IPC globals
// ---------------------------------------------------------------------------

static std::string g_simUuid;
static sem_t* g_semMetricsReady = nullptr;
static sem_t* g_semControl = nullptr;
static Ptr<MacScheduler> g_schedulerPtr = nullptr;
static Ptr<FlowMonitor> g_monitorPtr = nullptr;
static Ptr<Ipv4FlowClassifier> g_classifierPtr = nullptr;
static std::map<uint16_t, uint16_t> g_portToUeId;
static std::map<uint16_t, uint32_t> g_ueToSliceIdx;
static std::map<uint16_t, uint16_t> g_ueIdToRnti;
static Ipv4Address g_remoteHostAddr;
static uint32_t g_indicationPeriodMs = 100;
static bool g_ipcEnabled = false;
static bool g_enablePosixSync = false;
static bool g_enableDrlControl = true;
static std::string g_lastDrlActionStatus = "not_started";

static const std::vector<MacScheduler::IntraSliceAlgorithm> g_defaultSliceAlgorithms = {
    MacScheduler::IntraSliceAlgorithm::PF,
    MacScheduler::IntraSliceAlgorithm::RR,
    MacScheduler::IntraSliceAlgorithm::PF
};

struct KpmPrevValues
{
    uint64_t txBytes = 0;
    uint64_t rxBytes = 0;
    uint32_t txPackets = 0;
    uint32_t rxPackets = 0;
    uint32_t lostPackets = 0;
};
static std::map<uint16_t, KpmPrevValues> g_kpmPrevValues;
static std::map<uint16_t, KpmPrevValues> g_statsPrevValues;

struct GrSimState
{
    std::ofstream statsFile;
    std::ofstream macBufferFile;
    std::ofstream posixSyncLogFile;
    uint32_t stepCount{0};
    std::string outputDir;
    uint32_t indicationPeriodMs{100};
    double activeDurationSec{0.0};
};

static GrSimState g_simStats;

// ---------------------------------------------------------------------------
// Stats callback (timeseries CSV)
// ---------------------------------------------------------------------------

static void
StatsCallback()
{
    if (!g_monitorPtr || !g_classifierPtr)
    {
        Simulator::Schedule(MilliSeconds(g_simStats.indicationPeriodMs), &StatsCallback);
        return;
    }

    FlowMonitor::FlowStatsContainer stats = g_monitorPtr->GetFlowStats();
    uint64_t nowMs = static_cast<uint64_t>(Simulator::Now().GetMilliSeconds());
    g_simStats.stepCount++;

    if (!g_simStats.statsFile.is_open())
    {
        std::string path = g_simStats.outputDir + "/greenran_stats_timeseries.csv";
        g_simStats.statsFile.open(path, std::ios::out | std::ios::trunc);
        NS_ABORT_MSG_IF(!g_simStats.statsFile.is_open(), "Failed to open stats CSV: " << path);
        g_simStats.statsFile << "timestamp_ms,ue_id,slice_id,direction,thr_mbps,tx_bytes,plr_pct,effective_lost_pct,rsh_pct\n";
    }

    for (const auto& kv : stats)
    {
        Ipv4FlowClassifier::FiveTuple tuple = g_classifierPtr->FindFlow(kv.first);
        if (tuple.protocol != 17) continue;

        Direction direction = ClassifyDirection(tuple, g_remoteHostAddr);
        if (direction == Direction::Unknown) continue;

        auto itPort = g_portToUeId.find(tuple.destinationPort);
        if (itPort == g_portToUeId.end()) continue;

        uint16_t ueId = itPort->second;
        auto itSlice = g_ueToSliceIdx.find(ueId);
        if (itSlice == g_ueToSliceIdx.end()) continue;
        uint32_t sliceIdx = itSlice->second;
        const FlowMonitor::FlowStats& st = kv.second;

        KpmPrevValues& prev = g_statsPrevValues[ueId];
        uint64_t dRxBytes = (st.rxBytes >= prev.rxBytes) ? (st.rxBytes - prev.rxBytes) : 0;

        // Use cumulative TX/RX for PLR to avoid 10ms window artifact
        uint32_t cumulativeLost = (st.txPackets > st.rxPackets) ? (st.txPackets - st.rxPackets) : 0;
        double effectiveLostPct = (st.txPackets > 0) ? (static_cast<double>(cumulativeLost) / st.txPackets * 100.0) : 0.0;

        double periodSec = static_cast<double>(g_simStats.indicationPeriodMs) / 1000.0;
        double thrMbps = (periodSec > 0) ? (static_cast<double>(dRxBytes) * 8.0 / periodSec / 1e6) : 0.0;
        double plrPct = effectiveLostPct;

        double rshPct = 0.0;
        if (g_schedulerPtr)
        {
            rshPct = g_schedulerPtr->GetPrbWeight(sliceIdx) * 100.0;
        }

        g_simStats.statsFile << nowMs << "," << ueId << "," << sliceIdx << ","
                             << DirectionToString(direction) << ","
                             << std::fixed << std::setprecision(4) << thrMbps << ","
                             << dRxBytes << ","
                             << std::setprecision(2) << plrPct << ","
                             << effectiveLostPct << ","
                             << std::setprecision(2) << rshPct << "\n";

        prev.txBytes = st.txBytes;
        prev.rxBytes = st.rxBytes;
        prev.txPackets = st.txPackets;
        prev.rxPackets = st.rxPackets;
        prev.lostPackets = st.lostPackets;
    }

    g_simStats.statsFile.flush();
    Simulator::Schedule(MilliSeconds(g_simStats.indicationPeriodMs), &StatsCallback);
}

// ---------------------------------------------------------------------------
// IPC: KPM write + action read callback
// ---------------------------------------------------------------------------

static void
LogPosixSyncEvent(uint64_t timestampMs,
                  const std::string& eventName,
                  const std::string& status,
                  const std::string& details)
{
    if (!g_enablePosixSync || !g_simStats.posixSyncLogFile.is_open())
    {
        return;
    }

    g_simStats.posixSyncLogFile << timestampMs << ","
                                << eventName << ","
                                << status << ","
                                << details << "\n";
    g_simStats.posixSyncLogFile.flush();
}

static void
KpmAndControlCallback()
{
    if (!g_ipcEnabled || !g_enablePosixSync || !g_monitorPtr || !g_classifierPtr || !g_schedulerPtr)
    {
        Simulator::Schedule(MilliSeconds(g_indicationPeriodMs), &KpmAndControlCallback);
        return;
    }

    uint64_t nowMs = static_cast<uint64_t>(Simulator::Now().GetMilliSeconds());

    FlowMonitor::FlowStatsContainer stats = g_monitorPtr->GetFlowStats();

    struct UeKpm
    {
        uint64_t txBytes = 0;
        uint64_t rxBytes = 0;
        uint32_t txPackets = 0;
        uint32_t rxPackets = 0;
        uint32_t lostPackets = 0;
        double delaySumSec = 0.0;
        uint32_t count = 0;
    };

    std::map<uint16_t, UeKpm> ueKpms;

    for (const auto& kv : stats)
    {
        Ipv4FlowClassifier::FiveTuple tuple = g_classifierPtr->FindFlow(kv.first);
        if (tuple.protocol != 17) continue;
        if (ClassifyDirection(tuple, g_remoteHostAddr) == Direction::Unknown) continue;

        auto itPort = g_portToUeId.find(tuple.destinationPort);
        if (itPort == g_portToUeId.end()) continue;

        uint16_t ueId = itPort->second;
        UeKpm& k = ueKpms[ueId];
        k.txBytes += kv.second.txBytes;
        k.rxBytes += kv.second.rxBytes;
        k.txPackets += kv.second.txPackets;
        k.rxPackets += kv.second.rxPackets;
        k.lostPackets += kv.second.lostPackets;
        k.delaySumSec += kv.second.delaySum.GetSeconds();
        k.count++;
    }

    {
        std::ofstream kpmFile("rslaq-kpms.txt", std::ios::out | std::ios::trunc);
        kpmFile << "timestamp,ueImsi,sliceId,dTxBytes,dRxBytes,plr,resourceSharePct,dLostPackets,"
                << "throughputMbps,bufferBytes,dlBufferBytes,ulBufferBytes,macBufferBytes,"
                << "congestionPct,lastDrlActionStatus\n";

        for (const auto& kv : ueKpms)
        {
            uint16_t ueId = kv.first;
            const UeKpm& k = kv.second;
            auto itSlice = g_ueToSliceIdx.find(ueId);
            if (itSlice == g_ueToSliceIdx.end())
            {
                continue;
            }
            uint32_t sliceIdx = itSlice->second;

            KpmPrevValues& prev = g_kpmPrevValues[ueId];
            uint64_t dTxBytes = (k.txBytes >= prev.txBytes) ? (k.txBytes - prev.txBytes) : 0;
            uint64_t dRxBytes = (k.rxBytes >= prev.rxBytes) ? (k.rxBytes - prev.rxBytes) : 0;
            uint32_t dTxPackets = (k.txPackets >= prev.txPackets) ? (k.txPackets - prev.txPackets) : 0;
            uint32_t dRxPackets = (k.rxPackets >= prev.rxPackets) ? (k.rxPackets - prev.rxPackets) : 0;

            prev.txBytes = k.txBytes;
            prev.rxBytes = k.rxBytes;
            prev.txPackets = k.txPackets;
            prev.rxPackets = k.rxPackets;
            prev.lostPackets = k.lostPackets;

            uint32_t effectiveLost = (dTxPackets >= dRxPackets) ? (dTxPackets - dRxPackets) : 0;

            double plr = (dTxPackets > 0)
                             ? (static_cast<double>(effectiveLost) / dTxPackets * 100.0)
                             : 0.0;
            double rsh = (g_schedulerPtr)
                             ? g_schedulerPtr->GetPrbWeight(sliceIdx) * 100.0
                             : 0.0;
            double periodSec = static_cast<double>(g_indicationPeriodMs) / 1000.0;
            double thrMbps = (periodSec > 0)
                                 ? (static_cast<double>(dRxBytes) * 8.0 / periodSec / 1e6)
                                 : 0.0;

            // Directional buffer: UL for VIDEO_EMBB/SENSOR_MMTC, DL for GENERIC_EMBB
            uint32_t dlBufferBytes = (g_schedulerPtr && g_ueIdToRnti.count(ueId))
                                         ? g_schedulerPtr->GetUeDlBufferSize(g_ueIdToRnti[ueId])
                                         : 0;
            uint32_t ulBufferBytes = (g_schedulerPtr && g_ueIdToRnti.count(ueId))
                                         ? g_schedulerPtr->GetUeUlBufferSize(g_ueIdToRnti[ueId])
                                         : 0;
            uint32_t bufferBytes = (sliceIdx == GENERIC_EMBB_SLICE) ? dlBufferBytes : ulBufferBytes;
            uint32_t macBufferBytes = dlBufferBytes + ulBufferBytes;
            double congestionPct = std::min(100.0, static_cast<double>(macBufferBytes) / 10485760.0 * 100.0);

            kpmFile << nowMs << "," << ueId << "," << sliceIdx << ","
                    << dTxBytes << "," << dRxBytes << ","
                    << std::fixed << std::setprecision(2) << plr << ","
                    << std::setprecision(2) << rsh << ","
                    << effectiveLost << ","
                    << std::setprecision(4) << thrMbps << ","
                    << bufferBytes << ","
                    << dlBufferBytes << ","
                    << ulBufferBytes << ","
                    << macBufferBytes << ","
                    << std::setprecision(2) << congestionPct << ","
                    << g_lastDrlActionStatus << "\n";
        }
        kpmFile.flush();
    }

    LogPosixSyncEvent(nowMs, "write_kpms", "success", "rslaq-kpms.txt written");

    if (g_semMetricsReady)
    {
        sem_post(g_semMetricsReady);
    }

    LogPosixSyncEvent(nowMs, "sem_post", "success", "g_semMetricsReady signaled");

    if (g_semControl)
    {
        int ret = sem_trywait(g_semControl);
        if (ret == 0)
        {
            LogPosixSyncEvent(nowMs, "sem_trywait", "success", "sem_control acknowledged");
            if (g_enableDrlControl)
            {
                DrlSliceAction action;
                if (TryParseDrlActionFile("rslaq_actions_for_ns3.csv", action))
                {
                    g_schedulerPtr->SetSliceConfiguration(action.weights, g_defaultSliceAlgorithms);
                    g_lastAppliedActionTimestamp = action.timestampMs;
                    g_lastDrlActionStatus = "applied";
                    std::ostringstream details;
                    details << "weights=" << WeightsToString(action.weights);
                    LogPosixSyncEvent(nowMs, "apply_drl_action", "success", details.str());
                }
                else
                {
                    g_lastDrlActionStatus = "no_valid_new_action";
                    LogPosixSyncEvent(nowMs,
                                      "apply_drl_action",
                                      "skipped",
                                      "rslaq_actions_for_ns3.csv missing, stale, or invalid");
                }
            }
        }
        else
        {
            g_lastDrlActionStatus = "waiting_for_action";
            LogPosixSyncEvent(nowMs, "sem_trywait", "not_ready", "no external confirmation");
        }
    }

    Simulator::Schedule(MilliSeconds(g_indicationPeriodMs), &KpmAndControlCallback);
}

static void
MacBufferAndCongestionCallback()
{
    if (!g_schedulerPtr)
    {
        Simulator::Schedule(MilliSeconds(g_simStats.indicationPeriodMs), &MacBufferAndCongestionCallback);
        return;
    }

    uint64_t nowMs = static_cast<uint64_t>(Simulator::Now().GetMilliSeconds());
    if (!g_simStats.macBufferFile.is_open())
    {
        std::string path = g_simStats.outputDir + "/greenran_mac_buffer_congestion.csv";
        g_simStats.macBufferFile.open(path, std::ios::out | std::ios::trunc);
        NS_ABORT_MSG_IF(!g_simStats.macBufferFile.is_open(), "Failed to open MAC buffer CSV: " << path);
        g_simStats.macBufferFile << "timestamp_ms,slice_id,slice_name,num_ues,dl_buffer_bytes,"
                                 << "ul_buffer_bytes,total_mac_buffer_bytes,avg_buffer_per_ue_bytes,"
                                 << "congestion_pct,prb_weight_pct,last_drl_action_status\n";
    }

    struct BufferAgg
    {
        uint32_t numUes = 0;
        uint64_t dl = 0;
        uint64_t ul = 0;
    };
    std::vector<BufferAgg> aggs(NUM_SLICES);
    for (const auto& kv : g_ueToSliceIdx)
    {
        uint16_t ueId = kv.first;
        uint32_t sliceId = kv.second;
        if (sliceId >= NUM_SLICES || !g_ueIdToRnti.count(ueId))
        {
            continue;
        }
        uint16_t rnti = g_ueIdToRnti[ueId];
        aggs[sliceId].numUes++;
        aggs[sliceId].dl += g_schedulerPtr->GetUeDlBufferSize(rnti);
        aggs[sliceId].ul += g_schedulerPtr->GetUeUlBufferSize(rnti);
    }

    for (uint32_t s = 0; s < NUM_SLICES; ++s)
    {
        uint64_t total = aggs[s].dl + aggs[s].ul;
        double avg = (aggs[s].numUes > 0) ? static_cast<double>(total) / aggs[s].numUes : 0.0;
        double congestionPct = std::min(100.0, static_cast<double>(total) / 10485760.0 * 100.0);
        double prbWeightPct = g_schedulerPtr->GetPrbWeight(s) * 100.0;
        g_simStats.macBufferFile << nowMs << ","
                                 << s << ","
                                 << SliceName(s) << ","
                                 << aggs[s].numUes << ","
                                 << aggs[s].dl << ","
                                 << aggs[s].ul << ","
                                 << total << ","
                                 << std::fixed << std::setprecision(2) << avg << ","
                                 << congestionPct << ","
                                 << prbWeightPct << ","
                                 << g_lastDrlActionStatus << "\n";
    }
    g_simStats.macBufferFile.flush();
    Simulator::Schedule(MilliSeconds(g_simStats.indicationPeriodMs), &MacBufferAndCongestionCallback);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int
main(int argc, char* argv[])
{
    auto grScenarios = InitGreenRanScenarios();

    std::string scenarioName = "greenran_normal";
    std::string outputDir = ".";
    uint32_t seed = 1;
    double simTimeSec = 4.0;
    std::string weightsStr = "";
    std::string simId = "";
    bool logAllMacSlots = false;
    uint32_t macLoggingPeriodMs = 97;
    bool enableUlSliceScheduling = true;
    uint32_t indicationPeriodMs = 100;

    uint32_t videoUes = 0;
    uint32_t sensorUes = 0;
    uint32_t genericUes = 0;
    double videoRateMbps = 0.0;
    double genericRateMbps = 0.0;
    uint32_t videoPacketSize = 0;
    uint32_t sensorPacketSize = 0;
    double sensorIntervalSec = 0.0;

    double centralFrequency = 3.55e9;
    double bandwidth = 100e6;
    uint16_t numerology = 1;
    double txPowerDbm = 37.0;

    std::string deploymentScenario = "umi";
    int enableShadowing = -1;
    double ueAreaRadius = -1.0;
    double gnbHeight = -1.0;
    double ueHeight = 1.5;

    std::string hardwareProfile = "ran650_aw3161_outdoor";
    std::string radioProfile = "ran650_n78";
    std::string antennaProfileName = "alpha_aw3161";
    std::string ran650PowerMode = "perPort";
    double antennaGainDbi = 17.4;
    double antennaDowntiltDeg = 6.0;
    double antennaAzimuthBeamwidthDeg = 65.0;
    double antennaElevationBeamwidthDeg = 7.0;
    double sectorAzimuthDeg = 0.0;
    bool forceIsotropicAntenna = false;
    bool enableHardwareValidation = true;
    bool allowOutOfDatasheetConfig = false;

    CommandLine cmd(__FILE__);
    cmd.AddValue("scenario", "GreenRAN scenario name", scenarioName);
    cmd.AddValue("simTime", "Total sim time (s)", simTimeSec);
    cmd.AddValue("seed", "RNG seed", seed);
    cmd.AddValue("outputDir", "Output directory", outputDir);
    cmd.AddValue("weights", "Optional override slice weights (VIDEO_EMBB,SENSOR_MMTC,GENERIC_EMBB)", weightsStr);
    cmd.AddValue("simId", "Simulation UUID for IPC semaphores", simId);
    cmd.AddValue("periodMs", "FlowMonitor stats/KPM period in ms (scientific CSV default: 100)", indicationPeriodMs);
    cmd.AddValue("LogAllMacSlots", "Log every MAC scheduling call", logAllMacSlots);
    cmd.AddValue("MacLoggingPeriodMs", "MAC logging period in ms", macLoggingPeriodMs);
    cmd.AddValue("EnableUlSliceScheduling", "Enable slice-aware UL scheduling", enableUlSliceScheduling);
    cmd.AddValue("centralFrequency", "Central frequency in Hz", centralFrequency);
    cmd.AddValue("bandwidth", "Bandwidth in Hz", bandwidth);
    cmd.AddValue("numerology", "NR numerology 0-4", numerology);
    cmd.AddValue("txPower", "Optional explicit gNB TX power override in dBm", txPowerDbm);
    cmd.AddValue("deploymentScenario", "Deployment config", deploymentScenario);
    cmd.AddValue("enableShadowing", "Shadowing: -1=default, 0=off, 1=on", enableShadowing);
    cmd.AddValue("ueAreaRadius", "UE distribution radius (m)", ueAreaRadius);
    cmd.AddValue("gnbHeight", "gNB height (m)", gnbHeight);
    cmd.AddValue("ueHeight", "UE height (m)", ueHeight);
    cmd.AddValue("videoUes", "Override num VIDEO_EMBB UEs", videoUes);
    cmd.AddValue("sensorUes", "Override num SENSOR_MMTC UEs", sensorUes);
    cmd.AddValue("genericUes", "Override num GENERIC_EMBB UEs", genericUes);
    cmd.AddValue("videoRateMbps", "Override video UL rate (Mbps)", videoRateMbps);
    cmd.AddValue("genericRateMbps", "Override generic streaming rate (Mbps)", genericRateMbps);
    cmd.AddValue("videoPacketSize", "Override video packet size (B)", videoPacketSize);
    cmd.AddValue("sensorPacketSize", "Override sensor packet size (B)", sensorPacketSize);
    cmd.AddValue("sensorIntervalSec", "Override sensor interval (s)", sensorIntervalSec);
    cmd.AddValue("hardwareProfile", "Hardware preset: ran650_aw3161_outdoor or liteon_flexfi_indoor", hardwareProfile);
    cmd.AddValue("radioProfile", "Radio profile: ran650_n78 or liteon_flexfi_fr1", radioProfile);
    cmd.AddValue("antennaProfile", "Antenna profile: alpha_aw3161 or internal", antennaProfileName);
    cmd.AddValue("ran650PowerMode", "RAN650 power mode: perPort or aggregate", ran650PowerMode);
    cmd.AddValue("antennaGainDbi", "Antenna gain in dBi for reporting/model metadata", antennaGainDbi);
    cmd.AddValue("antennaDowntiltDeg", "Electrical downtilt in degrees", antennaDowntiltDeg);
    cmd.AddValue("antennaAzimuthBeamwidthDeg", "Antenna azimuth beamwidth in degrees", antennaAzimuthBeamwidthDeg);
    cmd.AddValue("antennaElevationBeamwidthDeg", "Antenna elevation beamwidth in degrees", antennaElevationBeamwidthDeg);
    cmd.AddValue("sectorAzimuthDeg", "Sector azimuth/bearing in degrees", sectorAzimuthDeg);
    cmd.AddValue("forceIsotropicAntenna", "Use isotropic element even when ThreeGppAntennaModel is available", forceIsotropicAntenna);
    cmd.AddValue("enableHardwareValidation", "Validate radio/antenna datasheet constraints", enableHardwareValidation);
    cmd.AddValue("allowOutOfDatasheetConfig", "Warn instead of aborting on datasheet violations", allowOutOfDatasheetConfig);
    bool enablePosixSync = true;
    bool enableDrlControl = true;
    uint32_t posixWaitTimeoutMs = 0;
    cmd.AddValue("enablePosixSync", "Enable POSIX semaphore synchronization", enablePosixSync);
    cmd.AddValue("enableDrlControl", "Enable non-blocking external DRL control of slice PRB weights", enableDrlControl);
    cmd.AddValue("posixWaitTimeoutMs", "Deprecated; IPC uses non-blocking sem_trywait", posixWaitTimeoutMs);
    bool weightsOverrideProvided = CliArgProvided(argc, argv, "weights");
    bool txPowerOverrideProvided = CliArgProvided(argc, argv, "txPower");
    cmd.Parse(argc, argv);

    EnsureDirectoryExists(outputDir);

    if (enablePosixSync && simId.empty())
    {
        simId = "greenran_" + scenarioName + "_" + std::to_string(seed);
    }
    g_enablePosixSync = enablePosixSync;
    g_enableDrlControl = enableDrlControl;
    // Resolve scenario parameters
    double videoRateMbpsVal = 15.0;
    double genericRateMbpsVal = 5.0;
    uint32_t videoPacketSizeVal = 1400;
    uint32_t sensorPacketSizeVal = 100;
    double sensorIntervalSecVal = 10.0;
    double videoDlFeedbackRateKbps = 64.0;
    uint32_t videoUesVal = 5;
    uint32_t sensorUesVal = 8;
    uint32_t genericUesVal = 20;

    if (grScenarios.find(scenarioName) != grScenarios.end())
    {
        auto& sc = grScenarios[scenarioName];
        videoUesVal = sc.videoUes;
        sensorUesVal = sc.sensorUes;
        genericUesVal = sc.genericUes;
        videoRateMbpsVal = sc.videoUlRateMbps;
        genericRateMbpsVal = sc.genericDlRateMbps;
        videoPacketSizeVal = sc.videoPktSize;
        sensorPacketSizeVal = sc.sensorPktSize;
        sensorIntervalSecVal = sc.sensorIntervalSec;
        videoDlFeedbackRateKbps = sc.videoDlFeedbackRateKbps;
        if (!weightsOverrideProvided)
        {
            weightsStr = WeightsToString(sc.weights);
        }
    }
    else if (!weightsOverrideProvided)
    {
        weightsStr = "0.4,0.15,0.45";
    }

    if (videoUes > 0) videoUesVal = videoUes;
    if (sensorUes > 0) sensorUesVal = sensorUes;
    if (genericUes > 0) genericUesVal = genericUes;
    if (videoRateMbps > 0.0) videoRateMbpsVal = videoRateMbps;
    if (genericRateMbps > 0.0) genericRateMbpsVal = genericRateMbps;
    if (videoPacketSize > 0) videoPacketSizeVal = videoPacketSize;
    if (sensorPacketSize > 0) sensorPacketSizeVal = sensorPacketSize;
    if (sensorIntervalSec > 0.0) sensorIntervalSecVal = sensorIntervalSec;

    uint32_t numUeTotal = videoUesVal + sensorUesVal + genericUesVal;
    double appStartSec = 0.5;
    std::vector<ProjectSliceProfile> projectSlices =
        MakeUfpaGreenRanSliceProfiles(videoRateMbpsVal, videoDlFeedbackRateKbps, videoUesVal,
                                       genericRateMbpsVal, genericUesVal);

    std::vector<double> sliceWeights = ParseWeights(weightsStr);
    ValidateWeights(sliceWeights);
    const std::string weightsSource = weightsOverrideProvided ? "CLI" : "scenario";

    RadioUnitProfile radio = MakeRan650N78Profile();
    AntennaProfile antenna = MakeAlphaAw3161Profile(antennaDowntiltDeg);
    if (hardwareProfile == "ran650_aw3161_outdoor")
    {
        radio = MakeRan650N78Profile();
        antenna = MakeAlphaAw3161Profile(antennaDowntiltDeg);
        radioProfile = "ran650_n78";
        antennaProfileName = "alpha_aw3161";
    }
    else if (hardwareProfile == "liteon_flexfi_indoor")
    {
        radio = MakeLiteOnFlexFiIndoorProfile();
        antenna = {"LiteOn FlexFi internal antenna",
                   3.3e9,
                   3.8e9,
                   4,
                   antennaGainDbi,
                   antennaGainDbi,
                   360.0,
                   180.0,
                   0.0,
                   0.0,
                   0.0,
                   0.0,
                   0.0,
                   0.0,
                   "Internal",
                   "Indoor internal antenna",
                   "Indoor/IP20 small-cell profile; not used for outdoor reference results."};
        radioProfile = "liteon_flexfi_fr1";
        antennaProfileName = "internal";
        if (!txPowerOverrideProvided && ran650PowerMode == "perPort")
        {
            ran650PowerMode = "perPort";
        }
    }
    else
    {
        NS_FATAL_ERROR("Invalid hardwareProfile: " << hardwareProfile);
    }

    if (CliArgProvided(argc, argv, "antennaGainDbi"))
    {
        antenna.gainDbi = antennaGainDbi;
    }
    else
    {
        antennaGainDbi = antenna.gainDbi;
    }
    if (CliArgProvided(argc, argv, "antennaAzimuthBeamwidthDeg"))
    {
        antenna.azimuthBeamwidthDeg = antennaAzimuthBeamwidthDeg;
    }
    else
    {
        antennaAzimuthBeamwidthDeg = antenna.azimuthBeamwidthDeg;
    }
    if (CliArgProvided(argc, argv, "antennaElevationBeamwidthDeg"))
    {
        antenna.elevationBeamwidthDeg = antennaElevationBeamwidthDeg;
    }
    else
    {
        antennaElevationBeamwidthDeg = antenna.elevationBeamwidthDeg;
    }

    NS_ABORT_MSG_IF(antennaDowntiltDeg < antenna.minElectricalDowntiltDeg ||
                        antennaDowntiltDeg > antenna.maxElectricalDowntiltDeg,
                    "Antenna electrical downtilt " << antennaDowntiltDeg
                                                    << " deg is outside profile range ["
                                                    << antenna.minElectricalDowntiltDeg << ", "
                                                    << antenna.maxElectricalDowntiltDeg << "]");
    antenna.electricalDowntiltDeg = antennaDowntiltDeg;

    ValidateDatasheetConfig(radio,
                            antenna,
                            hardwareProfile,
                            deploymentScenario,
                            centralFrequency,
                            bandwidth,
                            numerology,
                            enableHardwareValidation,
                            allowOutOfDatasheetConfig);

    double configuredTxPowerDbm = txPowerDbm;
    std::string txPowerModeUsed = "explicit_cli";
    if (!txPowerOverrideProvided)
    {
        if (ran650PowerMode == "perPort")
        {
            configuredTxPowerDbm = radio.txPowerPerPortDbm;
            txPowerModeUsed = "perPort";
        }
        else if (ran650PowerMode == "aggregate")
        {
            configuredTxPowerDbm = radio.aggregateTxPowerDbm;
            txPowerModeUsed = "aggregate";
        }
        else
        {
            NS_FATAL_ERROR("Invalid ran650PowerMode: " << ran650PowerMode);
        }
    }
    else
    {
        std::cout << "WARNING: explicit --txPower overrides hardware profile power mode.\n";
    }

    NS_ABORT_MSG_IF(configuredTxPowerDbm < -20.0 || configuredTxPowerDbm > 60.0,
                    "Suspicious gNB TxPower " << configuredTxPowerDbm << " dBm");
    const double approximateEirpDbm = configuredTxPowerDbm + antenna.gainDbi;
    const double thermalNoiseDbm = ComputeThermalNoiseDbm(bandwidth, radio.receiverNoiseFigureDb);
    if (txPowerModeUsed == "aggregate")
    {
        std::cout << "WARNING: aggregate RAN650 power mode may overestimate coverage depending on 5G-LENA TxPower semantics.\n";
    }

    // Deployment config
    DeploymentConfig deployConfig = GetDeploymentConfig(deploymentScenario);
    double resolvedRadius = (ueAreaRadius > 0) ? ueAreaRadius : deployConfig.defaultRadiusM;
    double resolvedGnbHeight = (gnbHeight > 0) ? gnbHeight : deployConfig.defaultGnbHeightM;
    double resolvedUeHeight = (ueHeight > 0) ? ueHeight : deployConfig.defaultUeHeightM;
    bool shadowingEnabled = deployConfig.defaultShadowing;
    if (enableShadowing == 1) shadowingEnabled = true;
    else if (enableShadowing == 0) shadowingEnabled = false;

    NS_ABORT_MSG_IF(simTimeSec <= appStartSec, "simTime must be greater than appStartSec");
    NS_ABORT_MSG_IF(bandwidth <= 0.0, "Bandwidth must be > 0 Hz");
    NS_ABORT_MSG_IF(centralFrequency <= 0.0, "Central frequency must be > 0 Hz");
    NS_ABORT_MSG_IF(numerology > 4, "Numerology must be in [0,4]");
    uint32_t estimatedPrbs = EstimateNrPrbs(bandwidth, numerology);
    uint32_t estimatedRbPerRbg = EstimateRbsPerRbg(estimatedPrbs);
    uint32_t estimatedRbgs = (estimatedPrbs + estimatedRbPerRbg - 1) / estimatedRbPerRbg;
    NS_ABORT_MSG_IF(estimatedPrbs == 0, "Estimated NR PRB count is zero; check bandwidth/numerology");

    std::cout << "========================================\n"
              << "  GreenRAN 5G/NR Slice-Aware Simulation\n"
              << "========================================\n"
              << "Scenario      : " << scenarioName << "\n"
              << "UL slicing    : " << (enableUlSliceScheduling ? "ENABLED" : "DISABLED") << "\n"
              << "Video UEs     : " << videoUesVal << " (VIDEO_EMBB UL, " << videoRateMbpsVal << " Mbps each; DL feedback "
              << videoDlFeedbackRateKbps << " kbps each)\n"
              << "Sensor UEs    : " << sensorUesVal << " (SENSOR_MMTC, " << sensorPacketSizeVal << " B every " << sensorIntervalSecVal << "s)\n"
              << "Total UEs     : " << numUeTotal << "\n"
              << "Weights       : " << weightsStr << " (" << weightsSource << ")\n"
              << "Frequency     : " << centralFrequency / 1e9 << " GHz\n"
              << "Bandwidth     : " << bandwidth / 1e6 << " MHz\n"
              << "Numerology    : " << numerology << "\n"
              << "Est. PRB/RBG  : " << estimatedPrbs << " PRBs, " << estimatedRbgs
              << " RBGs, " << estimatedRbPerRbg << " RB/RBG\n"
              << "Hardware      : " << hardwareProfile << "\n"
              << "Radio Unit    : " << radio.name << " (" << radio.deployment << ", " << radio.ipRating
              << ", O-RAN split " << radio.oRanSplit << ")\n"
              << "Antenna Panel : " << antenna.name << " (" << antenna.ports << " ports, "
              << antenna.gainDbi << " dBi, " << antenna.azimuthBeamwidthDeg << "x"
              << antenna.elevationBeamwidthDeg << " deg, downtilt "
              << antenna.electricalDowntiltDeg << " deg)\n"
              << "RAN650 TX mode: " << txPowerModeUsed << "\n"
              << "TxPower set   : " << configuredTxPowerDbm << " dBm\n"
              << "Approx. EIRP  : " << approximateEirpDbm << " dBm (TxPower + antenna gain; documentation only)\n"
              << "Receiver NF   : " << radio.receiverNoiseFigureDb << " dB\n"
              << "Thermal noise : " << thermalNoiseDbm << " dBm over " << bandwidth / 1e6 << " MHz\n"
              << "Channel       : " << deployConfig.channelScenario << " (from " << deploymentScenario << ")\n"
              << "UE Area       : +/- " << resolvedRadius << " m\n"
              << "Antenna model : gNB 2x2 array represents 4 ports; element="
              << (forceIsotropicAntenna ? "IsotropicAntennaModel" : "ThreeGppAntennaModel")
              << "; AW3161 measured pattern file not loaded\n"
              << "Shadowing     : " << (shadowingEnabled ? "ENABLED" : "DISABLED") << "\n"
              << "Direction rule: DL if source IP == remoteHost; UL if destination IP == remoteHost\n"
              << "SimTime        : " << simTimeSec << "s\n"
              << "IPC enabled   : "
              << (enablePosixSync && !simId.empty() ? "YES (POSIX sync)" : "NO (standalone)") << "\n"
              << "DRL control   : " << (enableDrlControl ? "ENABLED" : "DISABLED")
              << " (actions: rslaq_actions_for_ns3.csv; KPM: rslaq-kpms.txt)\n"
              << "Output         : " << outputDir << "\n"
              << "========================================\n\n";

    // Setup IPC semaphores if simId provided
    if (!simId.empty() && enablePosixSync)
    {
        g_simUuid = simId;
        g_ipcEnabled = true;

        std::string semMetricsName = "/rslaq_metrics_" + simId;
        std::string semControlName = "/rslaq_control_" + simId;

        g_semMetricsReady = sem_open(semMetricsName.c_str(), O_CREAT, 0660, 0);
        g_semControl = sem_open(semControlName.c_str(), O_CREAT, 0660, 0);

        if (g_semMetricsReady == SEM_FAILED || g_semControl == SEM_FAILED)
        {
            std::cerr << "WARNING: Failed to create POSIX semaphores. Running standalone.\n";
            g_ipcEnabled = false;
            if (g_semMetricsReady != SEM_FAILED) sem_close(g_semMetricsReady);
            if (g_semControl != SEM_FAILED) sem_close(g_semControl);
            g_semMetricsReady = nullptr;
            g_semControl = nullptr;
        }
        else
        {
            std::cout << "[GreenRAN] POSIX semaphores created: "
                      << semMetricsName << " and " << semControlName
                      << " (non-blocking control check; deprecated timeout arg="
                      << posixWaitTimeoutMs << " ms)\n";
        }
    }

    g_indicationPeriodMs = indicationPeriodMs;

    RngSeedManager::SetSeed(seed);
    RngSeedManager::SetRun(1);
    Config::SetDefault("ns3::NrRlcUm::MaxTxBufferSize", UintegerValue(10485760));

    // ---- Nodes ----
    NodeContainer gNbNodes;
    gNbNodes.Create(1);
    NodeContainer ueNodes;
    ueNodes.Create(numUeTotal);
    NodeContainer remoteHostContainer;
    remoteHostContainer.Create(1);

    // ---- Mobility ----
    MobilityHelper mobility;
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    {
        Ptr<ListPositionAllocator> gnbPos = CreateObject<ListPositionAllocator>();
        gnbPos->Add(Vector(0.0, 0.0, resolvedGnbHeight));
        mobility.SetPositionAllocator(gnbPos);
        mobility.Install(gNbNodes);
    }
    {
        Ptr<RandomRectanglePositionAllocator> uePos = CreateObject<RandomRectanglePositionAllocator>();
        uePos->SetAttribute("X", StringValue("ns3::UniformRandomVariable[Min=-" + std::to_string(static_cast<int>(resolvedRadius)) + "|Max=" + std::to_string(static_cast<int>(resolvedRadius)) + "]"));
        uePos->SetAttribute("Y", StringValue("ns3::UniformRandomVariable[Min=-" + std::to_string(static_cast<int>(resolvedRadius)) + "|Max=" + std::to_string(static_cast<int>(resolvedRadius)) + "]"));
        MobilityHelper ueMob;
        ueMob.SetPositionAllocator(uePos);
        ueMob.SetMobilityModel("ns3::ConstantPositionMobilityModel");
        ueMob.Install(ueNodes);
        for (uint32_t i = 0; i < ueNodes.GetN(); ++i)
        {
            Ptr<MobilityModel> mob = ueNodes.Get(i)->GetObject<MobilityModel>();
            Vector v = mob->GetPosition();
            v.z = resolvedUeHeight;
            mob->SetPosition(v);
        }
    }
    {
        MobilityHelper rhMob;
        rhMob.SetMobilityModel("ns3::ConstantPositionMobilityModel");
        rhMob.Install(remoteHostContainer);
    }

    // ---- NR Helpers ----
    Ptr<NrPointToPointEpcHelper> epcHelper = CreateObject<NrPointToPointEpcHelper>();
    Ptr<NrHelper> nrHelper = CreateObject<NrHelper>();
    nrHelper->SetEpcHelper(epcHelper);
    nrHelper->SetSchedulerTypeId(MacScheduler::GetTypeId());
    nrHelper->SetSchedulerAttribute("EnableUlSliceScheduling", BooleanValue(enableUlSliceScheduling));

    nrHelper->SetUeAntennaAttribute("NumRows", UintegerValue(1));
    nrHelper->SetUeAntennaAttribute("NumColumns", UintegerValue(1));
    nrHelper->SetUeAntennaAttribute("AntennaElement", PointerValue(CreateObject<IsotropicAntennaModel>()));
    // The RAN650/AW3161 hardware has 4 RF/antenna ports. Represent that as
    // a 2x2 gNB antenna array in 5G-LENA. Do not model this as a 4x4
    // 16-element array, which would imply a different panel than the datasheet.
    nrHelper->SetGnbAntennaAttribute("NumRows", UintegerValue(2));
    nrHelper->SetGnbAntennaAttribute("NumColumns", UintegerValue(2));
    nrHelper->SetGnbAntennaAttribute("DowntiltAngle", DoubleValue(antenna.electricalDowntiltDeg * M_PI / 180.0));
    nrHelper->SetGnbAntennaAttribute("BearingAngle", DoubleValue(sectorAzimuthDeg * M_PI / 180.0));
    if (forceIsotropicAntenna)
    {
        nrHelper->SetGnbAntennaAttribute("AntennaElement", PointerValue(CreateObject<IsotropicAntennaModel>()));
    }
    else
    {
        nrHelper->SetGnbAntennaAttribute("AntennaElement", PointerValue(CreateObject<ThreeGppAntennaModel>()));
    }

    CcBwpCreator ccBwpCreator;
    CcBwpCreator::SimpleOperationBandConf bandConf(centralFrequency, bandwidth, numerology);
    bandConf.m_numBwp = 1;
    OperationBandInfo band = ccBwpCreator.CreateOperationBandContiguousCc(bandConf);

    Ptr<NrChannelHelper> channelHelper = CreateObject<NrChannelHelper>();
    channelHelper->ConfigureFactories(deployConfig.channelScenario, "Default", "ThreeGpp");
    channelHelper->SetChannelConditionModelAttribute("UpdatePeriod", TimeValue(MilliSeconds(0)));
    if (enableShadowing >= 0)
    {
        channelHelper->SetPathlossAttribute("ShadowingEnabled", BooleanValue(enableShadowing != 0));
    }
    else
    {
        channelHelper->SetPathlossAttribute("ShadowingEnabled", BooleanValue(shadowingEnabled));
    }
    channelHelper->AssignChannelsToBands({band});
    BandwidthPartInfoPtrVector allBwps = CcBwpCreator::GetAllBwps({band});

    InternetStackHelper internet;
    internet.Install(remoteHostContainer);
    internet.Install(ueNodes);

    NetDeviceContainer gnbNetDev = nrHelper->InstallGnbDevice(gNbNodes, allBwps);
    NetDeviceContainer ueNetDev = nrHelper->InstallUeDevice(ueNodes, allBwps);

    int64_t randomStream = seed;
    randomStream += nrHelper->AssignStreams(gnbNetDev, randomStream);
    randomStream += nrHelper->AssignStreams(ueNetDev, randomStream);

    nrHelper->GetGnbPhy(gnbNetDev.Get(0), 0)->SetAttribute("Numerology", UintegerValue(numerology));
    // Do not manually add antenna gain to TxPower if the selected antenna model
    // already applies gain internally. EIRP is reported only for documentation.
    nrHelper->GetGnbPhy(gnbNetDev.Get(0), 0)->SetAttribute("TxPower", DoubleValue(configuredTxPowerDbm));

    Ptr<Node> pgw = epcHelper->GetPgwNode();
    PointToPointHelper p2ph;
    p2ph.SetDeviceAttribute("DataRate", DataRateValue(DataRate("100Gbps")));
    p2ph.SetDeviceAttribute("Mtu", UintegerValue(2500));
    p2ph.SetChannelAttribute("Delay", TimeValue(Seconds(0.001)));
    NetDeviceContainer internetDevices = p2ph.Install(pgw, remoteHostContainer.Get(0));

    Ipv4AddressHelper ipv4h;
    ipv4h.SetBase("1.0.0.0", "255.0.0.0");
    Ipv4InterfaceContainer internetIpIfaces = ipv4h.Assign(internetDevices);

    Ipv4StaticRoutingHelper ipv4RoutingHelper;
    Ptr<Ipv4StaticRouting> remoteStatic = ipv4RoutingHelper.GetStaticRouting(remoteHostContainer.Get(0)->GetObject<Ipv4>());
    remoteStatic->AddNetworkRouteTo(Ipv4Address("7.0.0.0"), Ipv4Mask("255.0.0.0"), 1);
    Ipv4Address remoteHostAddr = internetIpIfaces.GetAddress(1);
    g_remoteHostAddr = remoteHostAddr;

    Ipv4InterfaceContainer ueIpIfaces = epcHelper->AssignUeIpv4Address(NetDeviceContainer(ueNetDev));
    for (uint32_t i = 0; i < ueNodes.GetN(); ++i)
    {
        Ptr<Ipv4StaticRouting> ueStatic = ipv4RoutingHelper.GetStaticRouting(ueNodes.Get(i)->GetObject<Ipv4>());
        ueStatic->SetDefaultRoute(epcHelper->GetUeDefaultGatewayAddress(), 1);
    }

    // Approximate a service-aware path by binding each UE's application slice
    // to a 5G-LENA QoS Flow/5QI before attach. This is still EPC/RAN-side and
    // does not instantiate NSSF/AMF/SMF/UPF network slicing.
    {
        std::string qosPath = outputDir;
        if (!qosPath.empty() && qosPath.back() != '/') qosPath += '/';
        qosPath += scenarioName + "_slice_qos_mapping.csv";
        std::ofstream qosCsv(qosPath, std::ios::out | std::ios::trunc);
        NS_ABORT_MSG_IF(!qosCsv.is_open(), "Failed to open QoS mapping CSV: " << qosPath);
        qosCsv << "ue_id,ue_ipv4,slice_id,slice_name,application,service_type,primary_direction,five_qi,qfi,rule_scope\n";

        std::cout << "\n=== UFPA GreenRAN Slice/QoS Approximation ===\n"
                  << "Source: projetoUFPA/19175_Proposta_Ajustada (4).md\n"
                  << "Scope: application traffic + NR QoS Flow/5QI + RAN MAC PRB slicing; not full 5GC slicing.\n";
        for (uint32_t i = 0; i < numUeTotal; ++i)
        {
            uint16_t ueId = static_cast<uint16_t>(i + 1);
            uint32_t sliceIdx;
            if (i < videoUesVal) sliceIdx = VIDEO_EMBB_SLICE;
            else if (i < videoUesVal + sensorUesVal) sliceIdx = SENSOR_MMTC_SLICE;
            else sliceIdx = GENERIC_EMBB_SLICE;

            const ProjectSliceProfile& profile = projectSlices.at(sliceIdx);
            Ipv4Address ueAddr = ueIpIfaces.GetAddress(i);

            // Primary QoS flow (per-slice default)
            NrQosFlow qosFlow(profile.fiveQi);
            Ptr<NrQosRule> qosRule =
                MakeUeAddressScopedQosRule(remoteHostAddr, ueAddr, static_cast<uint8_t>(20 + sliceIdx));
            uint8_t qfi = nrHelper->ActivateDedicatedQosFlow(ueNetDev.Get(i), qosFlow, qosRule);

            qosCsv << ueId << "," << ueAddr << "," << sliceIdx << "," << profile.sliceName << ","
                   << profile.applicationName << "," << profile.serviceType << ","
                   << profile.primaryDirection << "," << FiveQiToNumber(profile.fiveQi) << ","
                   << static_cast<uint32_t>(qfi) << ",ue_address_bidirectional\n";
            std::cout << "UE " << std::setw(3) << ueId << " -> " << profile.sliceName
                      << " app=" << profile.applicationName
                      << " 5QI=" << FiveQiToNumber(profile.fiveQi)
                      << " QFI=" << static_cast<uint32_t>(qfi)
                      << " rule=UE-address scoped\n";

            // Generic UEs: second QoS flow for VoIP (GBR voice)
            if (sliceIdx == GENERIC_EMBB_SLICE)
            {
                NrQosFlow qosFlowVoip(NrQosFlow::GBR_CONV_VOICE);
                Ptr<NrQosRule> qosRuleVoip =
                    MakeUeAddressScopedQosRule(remoteHostAddr, ueAddr, static_cast<uint8_t>(40 + sliceIdx));
                uint8_t qfi2 = nrHelper->ActivateDedicatedQosFlow(ueNetDev.Get(i), qosFlowVoip, qosRuleVoip);
                qosCsv << ueId << "," << ueAddr << "," << sliceIdx << "," << profile.sliceName << ","
                       << "VoIP" << "," << "GBR voice" << ",UL,"
                       << FiveQiToNumber(NrQosFlow::GBR_CONV_VOICE) << ","
                       << static_cast<uint32_t>(qfi2) << ",ue_address_bidirectional\n";
                std::cout << "UE " << std::setw(3) << ueId << " -> " << profile.sliceName
                          << " app=VoIP 5QI=" << FiveQiToNumber(NrQosFlow::GBR_CONV_VOICE)
                          << " QFI=" << static_cast<uint32_t>(qfi2)
                          << " rule=UE-address scoped\n";
            }
        }
        qosCsv.close();
        std::cout << "[GreenRAN] Slice/QoS mapping written to: " << qosPath << "\n";
    }

    // ---- Attach all UEs immediately (5G-LENA does not support dynamic attach safely) ----
    nrHelper->AttachToClosestGnb(ueNetDev, gnbNetDev);
    std::cout << "[GreenRAN] Attached all " << numUeTotal << " UEs\n";

    // ---- Configure scheduler ----
    Ptr<NrMacScheduler> schedBase = NrHelper::GetScheduler(gnbNetDev.Get(0), 0);
    Ptr<MacScheduler> scheduler = DynamicCast<MacScheduler>(schedBase);
    NS_ASSERT_MSG(scheduler != nullptr, "Failed to cast to MacScheduler");
    g_schedulerPtr = scheduler;

    scheduler->SetScenarioName(scenarioName);
    scheduler->SetOutputDir(outputDir);
    scheduler->SetAttribute("EnableDetailedMacLogging", BooleanValue(true));
    if (logAllMacSlots)
    {
        scheduler->SetAttribute("LogAllMacSlots", BooleanValue(true));
    }
    scheduler->SetAttribute("MacLoggingPeriodMs", UintegerValue(macLoggingPeriodMs));

    // ---- Slice mapping (after attach) ----
    double mappingTime = std::min(0.1, appStartSec - 0.05);
    if (mappingTime < 0) mappingTime = 0.05;

    std::map<uint16_t, uint16_t> portToUeId;

    Simulator::Schedule(Seconds(mappingTime), [scheduler, ueNetDev, &sliceWeights, &outputDir, &scenarioName, videoUesVal, sensorUesVal, genericUesVal, numUeTotal, &portToUeId]() {
        std::vector<std::vector<uint32_t>> sliceRntis(NUM_SLICES);
        std::cout << "\n=== GreenRAN UE Mapping (RNTI) ===\n"
                  << std::setw(4) << "Idx" << " | "
                  << std::setw(6) << "IMSI" << " | "
                  << std::setw(6) << "RNTI" << " | "
                  << std::setw(8) << "Slice" << "\n";

        for (uint32_t i = 0; i < numUeTotal; ++i)
        {
            uint16_t ueId = static_cast<uint16_t>(i + 1);
            uint32_t sliceIdx;
            if (i < videoUesVal) sliceIdx = VIDEO_EMBB_SLICE;
            else if (i < videoUesVal + sensorUesVal) sliceIdx = SENSOR_MMTC_SLICE;
            else sliceIdx = GENERIC_EMBB_SLICE;

            Ptr<NrUeNetDevice> ueDev = DynamicCast<NrUeNetDevice>(ueNetDev.Get(i));
            uint16_t rnti = UINT16_MAX;
            if (ueDev && ueDev->GetRrc())
            {
                rnti = ueDev->GetRrc()->GetRnti();
            }
            if (rnti == UINT16_MAX || rnti == 0)
            {
                std::cerr << "WARNING: UE " << ueId << " no valid RNTI!\n";
            }

            sliceRntis[sliceIdx].push_back(rnti);

            std::string sn = SliceName(sliceIdx);
            std::cout << std::setw(4) << ueId << " | "
                      << std::setw(6) << (i + 1) << " | "
                      << std::setw(6) << rnti << " | "
                      << std::setw(8) << sn << "\n";
        }

        scheduler->SetSliceUeMapping(NUM_SLICES, sliceRntis);

        std::vector<MacScheduler::IntraSliceAlgorithm> algos = {
            MacScheduler::IntraSliceAlgorithm::PF,
            MacScheduler::IntraSliceAlgorithm::RR,
            MacScheduler::IntraSliceAlgorithm::PF
        };
        scheduler->SetSliceConfiguration(sliceWeights, algos);

        std::string prefix = outputDir;
        if (!prefix.empty() && prefix.back() != '/') prefix += '/';
        if (!scenarioName.empty()) prefix += scenarioName + "_";
        std::ofstream mappingFile(prefix + "ue_rnti_mapping.csv", std::ios::out | std::ios::trunc);
        NS_ABORT_MSG_IF(!mappingFile.is_open(), "Failed to open UE/RNTI mapping CSV: " << prefix + "ue_rnti_mapping.csv");
        mappingFile << "ue_id,imsi,rnti,slice_id,slice_name\n";
        for (uint32_t i = 0; i < numUeTotal; ++i)
        {
            uint16_t ueId = static_cast<uint16_t>(i + 1);
            uint32_t sliceIdx;
            if (i < videoUesVal) sliceIdx = VIDEO_EMBB_SLICE;
            else if (i < videoUesVal + sensorUesVal) sliceIdx = SENSOR_MMTC_SLICE;
            else sliceIdx = GENERIC_EMBB_SLICE;
            std::string sn = SliceName(sliceIdx);
            uint16_t rnti = 0;
            Ptr<NrUeNetDevice> ueDev = DynamicCast<NrUeNetDevice>(ueNetDev.Get(i));
            if (ueDev && ueDev->GetRrc()) rnti = ueDev->GetRrc()->GetRnti();
            mappingFile << ueId << "," << (i + 1) << "," << rnti << "," << sliceIdx << "," << sn << "\n";
        }
        mappingFile.close();
        std::cout << "UE/RNTI mapping written to: " << prefix << "ue_rnti_mapping.csv\n";
    });

    // ---- Applications ----
    ApplicationContainer serverApps;
    ApplicationContainer clientApps;
    ApplicationContainer staggeredSensorApps;
    double activeDuration = simTimeSec - appStartSec;

    for (uint32_t i = 0; i < numUeTotal; ++i)
    {
        uint16_t ueId = static_cast<uint16_t>(i + 1);
        uint32_t sliceIdx;
        if (i < videoUesVal) sliceIdx = VIDEO_EMBB_SLICE;
        else if (i < videoUesVal + sensorUesVal) sliceIdx = SENSOR_MMTC_SLICE;
        else sliceIdx = GENERIC_EMBB_SLICE;
        Ipv4Address ueAddr = ueIpIfaces.GetAddress(i);

        // UL endpoint
        uint16_t ulPort = 30000 + ueId;
        UdpServerHelper ulServer(ulPort);
        serverApps.Add(ulServer.Install(remoteHostContainer.Get(0)));

        if (sliceIdx == VIDEO_EMBB_SLICE)
        {
            // Video upload from camera (UE) to server (UL)
            OnOffHelper ulVideo("ns3::UdpSocketFactory", InetSocketAddress(remoteHostAddr, ulPort));
            ulVideo.SetConstantRate(DataRate(videoRateMbpsVal * 1e6), videoPacketSizeVal);
            clientApps.Add(ulVideo.Install(ueNodes.Get(i)));
        }
        else if (sliceIdx == SENSOR_MMTC_SLICE)
        {
            UdpClientHelper sensorClient(remoteHostAddr, ulPort);
            sensorClient.SetAttribute("MaxPackets",
                                      UintegerValue(static_cast<uint32_t>(std::ceil(activeDuration / sensorIntervalSecVal)) + 2));
            sensorClient.SetAttribute("Interval", TimeValue(Seconds(sensorIntervalSecVal)));
            sensorClient.SetAttribute("PacketSize", UintegerValue(sensorPacketSizeVal));
            ApplicationContainer app = sensorClient.Install(ueNodes.Get(i));
            double jitterWindowSec = std::min(sensorIntervalSecVal, std::max(0.0, activeDuration * 0.8));
            double phaseJitterSec = jitterWindowSec *
                                    (static_cast<double>((ueId * 37) % 1000) / 1000.0);
            app.Start(Seconds(appStartSec + phaseJitterSec));
            app.Stop(Seconds(simTimeSec));
            staggeredSensorApps.Add(app);
        }
        else // GENERIC_EMBB_SLICE
        {
            // VoIP UL traffic — starts at appStartSec to ensure gNB sets up
            // DL logical channels early (needed for DL data to be buffered).
            OnOffHelper ulVoip("ns3::UdpSocketFactory", InetSocketAddress(remoteHostAddr, ulPort));
            ulVoip.SetConstantRate(DataRate(64000), 160);
            clientApps.Add(ulVoip.Install(ueNodes.Get(i)));
        }

        portToUeId[ulPort] = ueId;

        uint16_t dlPort = 40000 + ueId;
        if (sliceIdx == VIDEO_EMBB_SLICE)
        {
            // Small DL feedback from server to camera
            UdpServerHelper dlServer(dlPort);
            serverApps.Add(dlServer.Install(ueNodes.Get(i)));

            OnOffHelper dlFeedback("ns3::UdpSocketFactory", InetSocketAddress(ueAddr, dlPort));
            dlFeedback.SetConstantRate(DataRate(videoDlFeedbackRateKbps * 1000.0), 100);
            clientApps.Add(dlFeedback.Install(remoteHostContainer.Get(0)));

            portToUeId[dlPort] = ueId;
        }
        else if (sliceIdx == GENERIC_EMBB_SLICE)
        {
            UdpServerHelper dlServer(dlPort);
            serverApps.Add(dlServer.Install(ueNodes.Get(i)));

            // Staggered entry for generic users: traffic starts at different times.
            // Window limited to 10% of active time so that even the last UE has
            // enough runway (>90% of simTime) to generate meaningful traffic.
            uint32_t genericIdx = i - (videoUesVal + sensorUesVal);
            double entryWindow = (simTimeSec - appStartSec) * 0.1;
            double genericStartTime = appStartSec + (entryWindow * genericIdx / std::max(1u, genericUesVal));

            OnOffHelper dlStream("ns3::UdpSocketFactory", InetSocketAddress(ueAddr, dlPort));
            dlStream.SetConstantRate(DataRate(genericRateMbpsVal * 1e6), 1400);
            ApplicationContainer streamApp = dlStream.Install(remoteHostContainer.Get(0));
            streamApp.Start(Seconds(genericStartTime));
            streamApp.Stop(Seconds(simTimeSec));
            clientApps.Add(streamApp);

            portToUeId[dlPort] = ueId;
        }

        std::cout << "UE " << std::setw(3) << ueId
                  << " [" << std::setw(10) << SliceName(sliceIdx) << "]"
                  << " UL_port=" << ulPort
                  << " UL_model=";
        if (sliceIdx == VIDEO_EMBB_SLICE)
        {
            std::cout << "video_upload"
                      << " UL_rate=" << std::fixed << std::setprecision(2) << videoRateMbpsVal << " Mbps"
                      << " DL_port=" << dlPort
                      << " DL_rate=" << std::setprecision(0) << videoDlFeedbackRateKbps << " kbps"
                      << " DL_pkt=100 B\n";
        }
        else if (sliceIdx == SENSOR_MMTC_SLICE)
        {
            const double estimatedSensorKbps = sensorPacketSizeVal * 8.0 / sensorIntervalSecVal / 1000.0;
            std::cout << "periodic_sensor"
                      << " interval=" << std::fixed << std::setprecision(2) << sensorIntervalSecVal << " s"
                      << " sensor_pkt=" << sensorPacketSizeVal << " B"
                      << " est_rate=" << std::setprecision(4) << estimatedSensorKbps << " kbps\n";
        }
        else
        {
            std::cout << "voip"
                      << " UL_rate=64 kbps"
                      << " DL_port=" << dlPort
                      << " DL_rate=" << std::setprecision(2) << genericRateMbpsVal << " Mbps"
                      << " DL_pkt=1400 B\n";
        }
    }

    g_portToUeId = portToUeId;
    for (uint32_t i = 0; i < numUeTotal; ++i)
    {
        uint16_t ueId = static_cast<uint16_t>(i + 1);
        uint32_t sliceIdx;
        if (i < videoUesVal) sliceIdx = VIDEO_EMBB_SLICE;
        else if (i < videoUesVal + sensorUesVal) sliceIdx = SENSOR_MMTC_SLICE;
        else sliceIdx = GENERIC_EMBB_SLICE;
        g_ueToSliceIdx[ueId] = sliceIdx;
    }

    {
        std::string configPath = outputDir;
        if (!configPath.empty() && configPath.back() != '/') configPath += '/';
        configPath += scenarioName + "_simulation_config.json";
        std::ofstream cfg(configPath, std::ios::out | std::ios::trunc);
        NS_ABORT_MSG_IF(!cfg.is_open(), "Failed to open simulation config JSON: " << configPath);
        cfg << "{\n"
            << "  \"scenarioName\": \"" << scenarioName << "\",\n"
            << "  \"seed\": " << seed << ",\n"
            << "  \"simTimeSec\": " << simTimeSec << ",\n"
            << "  \"appStartSec\": " << appStartSec << ",\n"
            << "  \"numUes\": {\"VIDEO_EMBB\": " << videoUesVal
            << ", \"SENSOR_MMTC\": " << sensorUesVal
            << ", \"GENERIC_EMBB\": " << genericUesVal << "},\n"
            << "  \"traffic\": {\n"
            << "    \"VIDEO_EMBB\": {\"primaryDirection\": \"UL\", \"ulThroughputMbpsPerUe\": "
            << videoRateMbpsVal << ", \"ulPacketBytes\": " << videoPacketSizeVal
            << ", \"dlFeedbackKbpsPerUe\": " << videoDlFeedbackRateKbps << "},\n"
            << "    \"SENSOR_MMTC\": {\"primaryDirection\": \"UL\", \"packetBytes\": "
            << sensorPacketSizeVal << ", \"intervalSecMean\": " << sensorIntervalSecVal
            << ", \"startPhaseJitter\": \"deterministic_uniform\", \"estimatedKbpsPerSensor\": "
            << (sensorPacketSizeVal * 8.0 / sensorIntervalSecVal / 1000.0) << "},\n"
            << "    \"GENERIC_EMBB\": {\"primaryDirection\": \"DL+UL\", \"dlThroughputMbpsPerUe\": "
            << genericRateMbpsVal << ", \"dlPacketBytes\": 1400"
            << ", \"ulVoipKbpsPerUe\": 64}\n"
            << "  },\n"
            << "  \"projectSliceModel\": {\n"
            << "    \"source\": \"projetoUFPA/19175_Proposta_Ajustada (4).md\",\n"
            << "    \"scope\": \"closest feasible approximation: application classes, NR QoS Flow/5QI, EPC bearer metadata, and RAN MAC PRB slicing\",\n"
            << "    \"slices\": [\n";
        for (uint32_t s = 0; s < projectSlices.size(); ++s)
        {
            const ProjectSliceProfile& p = projectSlices.at(s);
            cfg << "      {\"sliceId\": " << p.sliceId
                << ", \"sliceName\": \"" << p.sliceName
                << "\", \"application\": \"" << p.applicationName
                << "\", \"useCase\": \"" << p.projectUseCase
                << "\", \"serviceType\": \"" << p.serviceType
                << "\", \"primaryDirection\": \"" << p.primaryDirection
                << "\", \"fiveQi\": " << FiveQiToNumber(p.fiveQi)
                << ", \"sla\": {\"dlThroughputMbps\": " << p.dlThroughputSlaMbps
                << ", \"ulThroughputMbps\": " << p.ulThroughputSlaMbps
                << ", \"delayMs\": " << p.delaySlaMs
                << ", \"pdr\": " << p.pdrSla << "}, \"notes\": \"" << p.qosNotes << "\"}"
                << (s + 1 == projectSlices.size() ? "\n" : ",\n");
        }
        cfg << "    ]\n"
            << "  },\n"
            << "  \"directionClassification\": \"DL if sourceAddress == remoteHostAddr; UL if destinationAddress == remoteHostAddr\",\n"
            << "  \"hardwareProfile\": \"" << hardwareProfile << "\",\n"
            << "  \"radioUnit\": {\n"
            << "    \"name\": \"" << radio.name << "\",\n"
            << "    \"frequencyRangeHz\": [" << radio.minFrequencyHz << ", " << radio.maxFrequencyHz << "],\n"
            << "    \"instantaneousBandwidthHz\": " << radio.maxBandwidthHz << ",\n"
            << "    \"scsKHz\": " << radio.scsKHz << ",\n"
            << "    \"numerology\": " << radio.numerology << ",\n"
            << "    \"txPorts\": " << radio.txPorts << ",\n"
            << "    \"rxPorts\": " << radio.rxPorts << ",\n"
            << "    \"txPowerPerPortDbm\": " << radio.txPowerPerPortDbm << ",\n"
            << "    \"aggregateTxPowerDbm\": " << radio.aggregateTxPowerDbm << ",\n"
            << "    \"configuredTxPowerDbm\": " << configuredTxPowerDbm << ",\n"
            << "    \"txPowerMode\": \"" << txPowerModeUsed << "\",\n"
            << "    \"receiverNoiseFigureDb\": " << radio.receiverNoiseFigureDb << ",\n"
            << "    \"typicalPowerConsumptionW\": " << radio.typicalPowerConsumptionW << ",\n"
            << "    \"ipRating\": \"" << radio.ipRating << "\",\n"
            << "    \"oRanSplit\": \"" << radio.oRanSplit << "\"\n"
            << "  },\n"
            << "  \"antennaPanel\": {\n"
            << "    \"name\": \"" << antenna.name << "\",\n"
            << "    \"type\": \"" << antenna.antennaType << "\",\n"
            << "    \"ports\": " << antenna.ports << ",\n"
            << "    \"gainDbi\": " << antenna.gainDbi << ",\n"
            << "    \"maxGainDbi\": " << antenna.maxGainDbi << ",\n"
            << "    \"azimuthBeamwidthDeg\": " << antenna.azimuthBeamwidthDeg << ",\n"
            << "    \"elevationBeamwidthDeg\": " << antenna.elevationBeamwidthDeg << ",\n"
            << "    \"electricalDowntiltDeg\": " << antenna.electricalDowntiltDeg << ",\n"
            << "    \"polarization\": \"" << antenna.polarization << "\",\n"
            << "    \"frontToBackRatioDb\": " << antenna.frontToBackRatioDb << ",\n"
            << "    \"crossPolarDiscriminationDb\": " << antenna.crossPolarDiscriminationDb << ",\n"
            << "    \"isolationDb\": " << antenna.isolationDb << "\n"
            << "  },\n"
            << "  \"weights\": {\"source\": \"" << weightsSource << "\", \"values\": ["
            << sliceWeights[0] << ", " << sliceWeights[1] << ", " << sliceWeights[2] << "]},\n"
            << "  \"nr\": {\"centralFrequencyHz\": " << centralFrequency
            << ", \"bandwidthHz\": " << bandwidth
            << ", \"numerology\": " << numerology
            << ", \"estimatedPrbs\": " << estimatedPrbs
            << ", \"estimatedRbgs\": " << estimatedRbgs
            << ", \"estimatedRbPerRbg\": " << estimatedRbPerRbg
            << ", \"txPowerDbm\": " << configuredTxPowerDbm
            << ", \"approximateEirpDbm\": " << approximateEirpDbm
            << ", \"thermalNoiseDbm\": " << thermalNoiseDbm << "},\n"
            << "  \"channel\": {\"deploymentScenarioCli\": \"" << deploymentScenario
            << "\", \"threeGppScenario\": \"" << deployConfig.channelScenario
            << "\", \"shadowing\": " << (shadowingEnabled ? "true" : "false") << "},\n"
            << "  \"mobility\": {\"ueAreaRadiusM\": " << resolvedRadius
            << ", \"gnbHeightM\": " << resolvedGnbHeight
            << ", \"ueHeightM\": " << resolvedUeHeight << "},\n"
            << "  \"antenna\": {\"gnbRows\": 2, \"gnbColumns\": 2, \"ueRows\": 1, \"ueColumns\": 1,"
            << " \"element\": \"" << (forceIsotropicAntenna ? "IsotropicAntennaModel" : "ThreeGppAntennaModel")
            << "\", \"exactAw3161PatternLoaded\": false, \"mimoSpatialMultiplexingClaimed\": false},\n"
            << "  \"scheduler\": {\"type\": \"MacScheduler\", \"ranSideOnly\": true,"
            << " \"ulSlicingEnabled\": " << (enableUlSliceScheduling ? "true" : "false") << "},\n"
            << "  \"ipc\": {\"enabled\": " << (g_ipcEnabled ? "true" : "false")
            << ", \"drlControlEnabled\": " << (g_enableDrlControl ? "true" : "false")
            << ", \"mode\": \"non_blocking_sem_trywait\", \"simId\": \"" << simId
            << "\", \"metricsSemaphore\": \"/rslaq_metrics_" << simId
            << "\", \"controlSemaphore\": \"/rslaq_control_" << simId
            << "\", \"kpmFile\": \"rslaq-kpms.txt\", \"actionFile\": \"rslaq_actions_for_ns3.csv\","
            << " \"actionFormat\": \"timestamp,sliceId,dedicatedPRB,minPRB,maxPRB\"},\n"
            << "  \"kpmAndMacMetrics\": {\"macBufferCongestionCsv\": \"greenran_mac_buffer_congestion.csv\","
            << " \"slicePrbCsvPattern\": \"*_slice_alloc*.csv\", \"bufferCongestionReferenceBytes\": 10485760},\n"
            << "  \"modelingNotes\": {\"slicingScope\": \"near-end-to-end approximation with RAN/MAC enforcement by UE/RNTI\","
            << " \"notModeled\": [\"full 5GC slicing\", \"NSSF/AMF/SMF/UPF slicing\","
            << " \"real SDAP entity\", \"exact measured antenna pattern file\","
            << " \"real O-RAN fronthaul latency unless explicitly modeled\"]},\n"
            << "  \"modelLimitations\": [\"RAN-side MAC/RBG slicing only\", \"No complete 5GC slicing\","
            << " \"No NSSF/AMF/SMF/UPF slicing\", \"5QI/QFI are represented with 5G-LENA dedicated QoS flows but not a full 5GC policy chain\","
            << " \"AW3161 exact measured radiation pattern is not loaded\","
            << " \"Hardware profile changed; old results are not directly comparable\"]\n"
            << "}\n";
        cfg.close();
        std::cout << "[GreenRAN] Simulation config written to: " << configPath << "\n";

        std::string hardwareCsvPath = outputDir;
        if (!hardwareCsvPath.empty() && hardwareCsvPath.back() != '/') hardwareCsvPath += '/';
        hardwareCsvPath += scenarioName + "_hardware_profile.csv";
        std::ofstream hwCsv(hardwareCsvPath, std::ios::out | std::ios::trunc);
        NS_ABORT_MSG_IF(!hwCsv.is_open(), "Failed to open hardware profile CSV: " << hardwareCsvPath);
        hwCsv << "category,parameter,value,unit,source_or_note\n"
              << "radio,name," << radio.name << ",,datasheet\n"
              << "radio,rf_min," << radio.minFrequencyHz << ",Hz,datasheet\n"
              << "radio,rf_max," << radio.maxFrequencyHz << ",Hz,datasheet\n"
              << "radio,max_bandwidth," << radio.maxBandwidthHz << ",Hz,datasheet\n"
              << "radio,scs," << radio.scsKHz << ",kHz,datasheet\n"
              << "radio,numerology," << radio.numerology << ",,30 kHz SCS\n"
              << "radio,tx_ports," << radio.txPorts << ",,datasheet\n"
              << "radio,rx_ports," << radio.rxPorts << ",,datasheet\n"
              << "radio,tx_power_per_port," << radio.txPowerPerPortDbm << ",dBm,datasheet\n"
              << "radio,aggregate_tx_power," << radio.aggregateTxPowerDbm << ",dBm,37+10log10(4)\n"
              << "radio,configured_tx_power," << configuredTxPowerDbm << ",dBm,5G-LENA TxPower attribute\n"
              << "radio,receiver_noise_figure," << radio.receiverNoiseFigureDb << ",dB,datasheet\n"
              << "radio,typical_power_consumption," << radio.typicalPowerConsumptionW << ",W,datasheet\n"
              << "radio,ip_rating," << radio.ipRating << ",,datasheet\n"
              << "antenna,name," << antenna.name << ",,datasheet\n"
              << "antenna,type," << antenna.antennaType << ",,datasheet\n"
              << "antenna,ports," << antenna.ports << ",,datasheet\n"
              << "antenna,gain," << antenna.gainDbi << ",dBi,datasheet/default or CLI\n"
              << "antenna,max_gain," << antenna.maxGainDbi << ",dBi,datasheet\n"
              << "antenna,azimuth_beamwidth," << antenna.azimuthBeamwidthDeg << ",deg,datasheet/default or CLI\n"
              << "antenna,elevation_beamwidth," << antenna.elevationBeamwidthDeg << ",deg,datasheet/default or CLI\n"
              << "antenna,electrical_downtilt," << antenna.electricalDowntiltDeg << ",deg,CLI/default\n"
              << "antenna,polarization," << antenna.polarization << ",,datasheet\n"
              << "link_budget,approximate_eirp," << approximateEirpDbm << ",dBm,TxPower+AntennaGain documentation only\n"
              << "link_budget,thermal_noise," << thermalNoiseDbm << ",dBm,-174+10log10(BW)+NF\n"
              << "model,antenna_element," << (forceIsotropicAntenna ? "IsotropicAntennaModel" : "ThreeGppAntennaModel")
              << ",,5G-LENA model selection\n"
              << "model,exact_aw3161_pattern_loaded,false,,no measured pattern file supplied\n";
        hwCsv.close();
        std::cout << "[GreenRAN] Hardware profile CSV written to: " << hardwareCsvPath << "\n";

        std::string reportPath = outputDir;
        if (!reportPath.empty() && reportPath.back() != '/') reportPath += '/';
        reportPath += scenarioName + "_hardware_model_report.md";
        std::ofstream report(reportPath, std::ios::out | std::ios::trunc);
        NS_ABORT_MSG_IF(!report.is_open(), "Failed to open hardware model report: " << reportPath);
        report << "# GreenRAN Hardware Model Report\n\n"
               << "## Hardware modelado\n\n"
               << "- Radio: " << radio.name << "\n"
               << "- Antena: " << antenna.name << "\n"
               << "- Perfil: " << hardwareProfile << "\n"
               << "- Deployment: " << deploymentScenario << " / " << deployConfig.channelScenario << "\n\n"
               << "## Parametros implementados literalmente\n\n"
               << "- Faixa RF 3.3-3.8 GHz para n78.\n"
               << "- Bandwidth instantaneo maximo de 100 MHz.\n"
               << "- SCS 30 kHz / numerologia 1.\n"
               << "- RAN650 4T4R, 37 dBm por porta, potencia agregada calculada "
               << std::fixed << std::setprecision(2) << radio.aggregateTxPowerDbm << " dBm.\n"
               << "- Receiver noise figure " << radio.receiverNoiseFigureDb << " dB.\n"
               << "- IP rating " << radio.ipRating << " e consumo tipico " << radio.typicalPowerConsumptionW << " W.\n"
               << "- AW3161 com 4 portas, ganho " << antenna.gainDbi << " dBi, beamwidth "
               << antenna.azimuthBeamwidthDeg << "/" << antenna.elevationBeamwidthDeg
               << " graus, downtilt " << antenna.electricalDowntiltDeg << " graus, polarizacao "
               << antenna.polarization << ".\n\n"
               << "## Aproximacoes\n\n"
               << "- O padrao de radiacao medido da AW3161 nao e carregado; sem arquivo real de pattern, "
               << "beamwidth, front-to-back e XPD sao metadados rastreaveis.\n"
               << "- As 4 portas foram aproximadas como matriz gNB 2x2 no 5G-LENA, nao como 4x4/16 elementos.\n"
               << "- O calculo de EIRP (" << approximateEirpDbm << " dBm) e informativo; nao foi somado manualmente ao TxPower.\n"
               << "- O-RAN split " << radio.oRanSplit << " e documentado, mas latencia/jitter reais de fronthaul nao sao modelados.\n"
               << "- O slicing e aproximado por trafego de aplicacao, QoS Flow/5QI dedicado no 5G-LENA e enforcement RAN/MAC-side por UE/RNTI.\n\n"
               << "## Slice do projeto UFPA\n\n"
               << "- App1-Vigilancia: VIDEO_EMBB, video 4K/H.265 campus UFPA, 5QI "
               << FiveQiToNumber(projectSlices[VIDEO_EMBB_SLICE].fiveQi)
               << ", SLA DL " << projectSlices[VIDEO_EMBB_SLICE].dlThroughputSlaMbps
               << " Mbps agregado e delay < " << projectSlices[VIDEO_EMBB_SLICE].delaySlaMs << " ms.\n"
               << "- App2-Monitoramento: SENSOR_MMTC, sensores ambientais/solo, 5QI "
               << FiveQiToNumber(projectSlices[SENSOR_MMTC_SLICE].fiveQi)
               << ", trafego UL esporadico e foco em PDR/congestionamento/uso eficiente de PRBs.\n"
               << "- App3-Usuarios: GENERIC_EMBB, streaming + VoIP, 5QI "
               << FiveQiToNumber(projectSlices[GENERIC_EMBB_SLICE].fiveQi)
               << " (streaming) + 5QI 1 (VoIP), SLA DL " << projectSlices[GENERIC_EMBB_SLICE].dlThroughputSlaMbps
               << " Mbps agregado e delay < " << projectSlices[GENERIC_EMBB_SLICE].delaySlaMs << " ms.\n"
               << "- DRL/POSIX: semaforos POSIX ficam habilitados por padrao quando `enablePosixSync=true`; "
               << "o controle de pesos de PRB usa `rslaq_actions_for_ns3.csv` no formato "
               << "`timestamp,sliceId,dedicatedPRB,minPRB,maxPRB` e leitura nao bloqueante com `sem_trywait`.\n"
               << "- Resultados de PRB e buffer MAC sao exportados em `*_slice_alloc*.csv`, "
               << "`*_ue_detail*.csv`, `rslaq-kpms.txt` e `greenran_mac_buffer_congestion.csv`.\n\n"
               << "## Alertas\n\n"
               << "- Frequencia fora de 3.3-3.8 GHz, bandwidth acima de 100 MHz ou numerologia diferente de 1 abortam por padrao.\n"
               << "- `allowOutOfDatasheetConfig=true` troca abort por warning forte para estudos experimentais.\n"
               << "- `ran650PowerMode=aggregate` pode superestimar cobertura dependendo da semantica de TxPower do 5G-LENA.\n"
               << "- Se `forceIsotropicAntenna=true`, o relatorio nao deve ser interpretado como setor AW3161 fisico.\n"
               << "- Resultados antigos nao sao diretamente comparaveis quando o perfil de hardware/canal muda.\n\n"
               << "Este cenario representa uma aproximacao de uma instalacao outdoor com RAN650 n78 + antena painel AW3161. "
               << "Ele nao representa network slicing end-to-end completo nem o padrao de radiacao medido da antena, salvo se forem fornecidos e carregados arquivos reais de pattern.\n";
        report.close();
        std::cout << "[GreenRAN] Hardware model report written to: " << reportPath << "\n";
    }

    serverApps.Start(Seconds(appStartSec));
    clientApps.Start(Seconds(appStartSec));
    serverApps.Stop(Seconds(simTimeSec));
    clientApps.Stop(Seconds(simTimeSec));

    // ---- Flow Monitor ----
    FlowMonitorHelper flowmonHelper;
    NodeContainer endpointNodes;
    endpointNodes.Add(remoteHostContainer);
    endpointNodes.Add(ueNodes);
    Ptr<FlowMonitor> monitor = flowmonHelper.Install(endpointNodes);
    monitor->SetAttribute("DelayBinWidth", DoubleValue(0.001));
    monitor->SetAttribute("JitterBinWidth", DoubleValue(0.001));
    monitor->SetAttribute("PacketSizeBinWidth", DoubleValue(20));
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(flowmonHelper.GetClassifier());

    g_monitorPtr = monitor;
    g_classifierPtr = classifier;

    // Populate g_ueIdToRnti after attach
    Simulator::Schedule(Seconds(mappingTime + 0.01), [ueNetDev, numUeTotal]() {
        for (uint32_t i = 0; i < numUeTotal; ++i)
        {
            uint16_t ueId = static_cast<uint16_t>(i + 1);
            Ptr<NrUeNetDevice> ueDev = DynamicCast<NrUeNetDevice>(ueNetDev.Get(i));
            if (ueDev && ueDev->GetRrc())
            {
                g_ueIdToRnti[ueId] = ueDev->GetRrc()->GetRnti();
            }
        }
    });

    // ---- Schedule callbacks ----
    g_simStats.outputDir = outputDir;
    g_simStats.indicationPeriodMs = indicationPeriodMs;
    g_simStats.activeDurationSec = simTimeSec - appStartSec;

    Time firstCallback = Seconds(appStartSec) + MilliSeconds(indicationPeriodMs);
    Simulator::Schedule(firstCallback, &StatsCallback);
    Simulator::Schedule(firstCallback, &MacBufferAndCongestionCallback);

    if (g_ipcEnabled)
    {
        Simulator::Schedule(firstCallback, &KpmAndControlCallback);
        std::cout << "[GreenRAN] IPC callback scheduled at t=" << firstCallback.GetMilliSeconds() << " ms\n";
    }

    if (g_enablePosixSync)
    {
        std::string posixLogPath = outputDir;
        if (!posixLogPath.empty() && posixLogPath.back() != '/')
        {
            posixLogPath += '/';
        }
        posixLogPath += scenarioName + "_posix_sync_log.csv";
        g_simStats.posixSyncLogFile.open(posixLogPath, std::ios::out | std::ios::trunc);
        NS_ABORT_MSG_IF(!g_simStats.posixSyncLogFile.is_open(), "Failed to open POSIX sync log: " << posixLogPath);
        g_simStats.posixSyncLogFile << "timestamp_ms,event,status,details\n";
        g_simStats.posixSyncLogFile.flush();
        std::cout << "[GreenRAN] POSIX sync log: " << posixLogPath << "\n";
    }

    // ---- Run ----
    Simulator::Stop(Seconds(simTimeSec));
    Simulator::Run();

    // ---- Final stats ----
    if (g_simStats.statsFile.is_open())
    {
        g_simStats.statsFile.close();
    }
    if (g_simStats.macBufferFile.is_open())
    {
        g_simStats.macBufferFile.close();
    }

    if (g_simStats.posixSyncLogFile.is_open())
    {
        g_simStats.posixSyncLogFile.close();
    }

    monitor->CheckForLostPackets();
    FlowMonitor::FlowStatsContainer stats = monitor->GetFlowStats();

    std::string prefix = outputDir;
    if (!prefix.empty() && prefix.back() != '/') prefix += '/';
    if (!scenarioName.empty()) prefix += scenarioName + "_";

    const std::vector<FlowRecord> flowRecords =
        ExtractFlowRecords(stats, classifier, portToUeId, g_ueToSliceIdx, remoteHostAddr);

    struct DirAgg
    {
        uint64_t txB = 0;
        uint64_t rxB = 0;
        uint32_t txP = 0;
        uint32_t rxP = 0;
        uint32_t lostP = 0;
        double totalDelay = 0.0;
    };

    auto accumulateAgg = [](DirAgg& agg, const FlowRecord& record)
    {
        agg.txB += record.txBytes;
        agg.rxB += record.rxBytes;
        agg.txP += record.txPackets;
        agg.rxP += record.rxPackets;
        agg.lostP += record.lostPackets;
        agg.totalDelay += record.delaySumSec;
    };

    auto aggThroughput = [activeDuration](const DirAgg& agg)
    {
        return (activeDuration > 0) ? agg.rxB * 8.0 / activeDuration / 1e6 : 0.0;
    };

    auto aggDelayMs = [](const DirAgg& agg)
    {
        return (agg.rxP > 0) ? agg.totalDelay / agg.rxP * 1000.0 : 0.0;
    };

    auto aggPdr = [](const DirAgg& agg)
    {
        return (agg.txP > 0) ? static_cast<double>(agg.rxP) / agg.txP : 0.0;
    };

    auto aggEffectiveLost = [](const DirAgg& agg)
    {
        return (agg.txP > agg.rxP) ? agg.txP - agg.rxP : 0;
    };

    auto aggEffectivePdr = [&aggEffectiveLost](const DirAgg& agg)
    {
        return (agg.txP > 0) ? 1.0 - static_cast<double>(aggEffectiveLost(agg)) / agg.txP : 0.0;
    };

    std::map<uint32_t, std::map<Direction, DirAgg>> sliceAggs;
    for (const FlowRecord& record : flowRecords)
    {
        accumulateAgg(sliceAggs[record.sliceId][record.direction], record);
    }

    // Per-UE CSV
    {
        std::ofstream ueCsv(prefix + "greenran_ue.csv", std::ios::out | std::ios::trunc);
        NS_ABORT_MSG_IF(!ueCsv.is_open(), "Failed to open UE CSV: " << prefix + "greenran_ue.csv");
        ueCsv << "ue_id,slice_id,slice_name,direction,tx_bytes,rx_bytes,tx_packets,rx_packets,"
              << "lost_packets,effective_lost_packets,throughput_mbps,avg_delay_ms,pdr,effective_pdr\n";

        for (const FlowRecord& record : flowRecords)
        {
            uint16_t ueId = record.ueId;
            uint32_t sliceIdx = record.sliceId;
            std::string sn = SliceName(sliceIdx);
            double thr = (activeDuration > 0) ? record.rxBytes * 8.0 / activeDuration / 1e6 : 0.0;
            double avgDelay = (record.rxPackets > 0) ? record.delaySumSec / record.rxPackets * 1000.0 : 0.0;
            double pdr = (record.txPackets > 0) ? static_cast<double>(record.rxPackets) / record.txPackets : 0.0;
            uint32_t effLost = (record.txPackets > record.rxPackets) ? record.txPackets - record.rxPackets : 0;
            double effPdr = (record.txPackets > 0) ? 1.0 - static_cast<double>(effLost) / record.txPackets : 0.0;

            ueCsv << ueId << "," << sliceIdx << "," << sn << "," << DirectionToString(record.direction) << ","
                  << record.txBytes << "," << record.rxBytes << ","
                  << record.txPackets << "," << record.rxPackets << "," << record.lostPackets << ","
                  << effLost << "," << std::fixed << std::setprecision(4) << thr << ","
                  << std::setprecision(3) << avgDelay << ","
                  << std::setprecision(4) << pdr << "," << std::setprecision(4) << effPdr << "\n";
        }
    }

    // Per-slice CSV (aggregated per direction)
    {
        std::ofstream sliceCsv(prefix + "greenran_slice.csv", std::ios::out | std::ios::trunc);
        NS_ABORT_MSG_IF(!sliceCsv.is_open(), "Failed to open slice CSV: " << prefix + "greenran_slice.csv");
        sliceCsv << "slice_id,slice_name,num_ues,direction,tx_bytes,rx_bytes,tx_packets,rx_packets,"
                 << "lost_packets,effective_lost_packets,throughput_mbps,avg_delay_ms,pdr,effective_pdr\n";

        for (uint32_t s = 0; s < NUM_SLICES; ++s)
        {
            std::string sn = SliceName(s);
            uint32_t nUes = 0;
            if (s == VIDEO_EMBB_SLICE) nUes = videoUesVal;
            else if (s == SENSOR_MMTC_SLICE) nUes = sensorUesVal;
            else if (s == GENERIC_EMBB_SLICE) nUes = genericUesVal;

            auto writeAgg = [&](const DirAgg& agg, const std::string& dirStr)
            {
                double thr = aggThroughput(agg);
                double avgDelay = aggDelayMs(agg);
                double pdr = aggPdr(agg);
                uint32_t effLost = aggEffectiveLost(agg);
                double effPdr = aggEffectivePdr(agg);
                sliceCsv << s << "," << sn << "," << nUes << "," << dirStr << ","
                         << agg.txB << "," << agg.rxB << "," << agg.txP << "," << agg.rxP << ","
                         << agg.lostP << "," << effLost << "," << std::fixed << std::setprecision(4) << thr << ","
                         << std::setprecision(3) << avgDelay << ","
                         << std::setprecision(4) << pdr << "," << std::setprecision(4) << effPdr << "\n";
            };

            writeAgg(sliceAggs[s][Direction::UL], "UL");
            writeAgg(sliceAggs[s][Direction::DL], "DL");
        }
    }

    // SLA CSV (per direction)
    {
        std::ofstream slaCsv(prefix + "greenran_sla.csv", std::ios::out | std::ios::trunc);
        NS_ABORT_MSG_IF(!slaCsv.is_open(), "Failed to open SLA CSV: " << prefix + "greenran_sla.csv");
        slaCsv << "scenario,slice,direction,sla_delay_ms,measured_avg_delay_ms,"
               << "sla_pdr,measured_effective_pdr,sla_throughput_mbps,measured_throughput_mbps,"
               << "delay_violation,pdr_violation,throughput_violation,overall_violation\n";

        for (uint32_t s = 0; s < NUM_SLICES; ++s)
        {
            std::string sn = SliceName(s);
            const ProjectSliceProfile& profile = projectSlices.at(s);
            SliceSla sla = {profile.dlThroughputSlaMbps,
                            profile.ulThroughputSlaMbps,
                            profile.delaySlaMs,
                            profile.pdrSla};

            auto writeSla = [&](const DirAgg& agg, const std::string& dirStr)
            {
                double thr = aggThroughput(agg);
                double avgDelay = aggDelayMs(agg);
                double effPdr = aggEffectivePdr(agg);
                double slaTput = (dirStr == "DL") ? sla.dlThroughputMbps : sla.ulThroughputMbps;

                bool delayViol = (agg.rxP > 0) && (avgDelay > sla.delayMs);
                bool pdrViol = (agg.txP > 0) && (effPdr < sla.pdr);
                bool tputViol = (slaTput > 0) && (thr < slaTput);
                bool overallViol = delayViol || pdrViol || tputViol;

                slaCsv << scenarioName << "," << sn << "," << dirStr << ","
                       << std::fixed << std::setprecision(1) << sla.delayMs << ","
                       << std::setprecision(3) << avgDelay << ","
                       << std::setprecision(4) << sla.pdr << "," << effPdr << ","
                       << std::setprecision(2) << slaTput << "," << thr << ","
                       << (delayViol ? 1 : 0) << ","
                       << (pdrViol ? 1 : 0) << ","
                       << (tputViol ? 1 : 0) << ","
                       << (overallViol ? 1 : 0) << "\n";
            };

            writeSla(sliceAggs[s][Direction::UL], "UL");
            writeSla(sliceAggs[s][Direction::DL], "DL");
        }
    }

    // Print summary
    std::cout << "\n========================================\n"
              << "  GreenRAN Results (" << scenarioName << ")\n"
              << "========================================\n";

    for (uint32_t s = 0; s < NUM_SLICES; ++s)
    {
        std::string sn = SliceName(s);

        auto printAgg = [&](const DirAgg& agg, const std::string& dirStr)
        {
            double thr = aggThroughput(agg);
            double avgDelay = aggDelayMs(agg);
            double pdr = aggPdr(agg);
            std::cout << sn << " [" << dirStr << "]:\n"
                      << "  Throughput : " << std::fixed << std::setprecision(4) << thr << " Mbps\n"
                      << "  Avg delay  : " << std::setprecision(3) << avgDelay << " ms\n"
                      << "  PDR        : " << std::setprecision(4) << pdr << "\n"
                      << "  TX/RX pkts : " << agg.txP << " / " << agg.rxP << "\n\n";
        };
        printAgg(sliceAggs[s][Direction::UL], "UL");
        printAgg(sliceAggs[s][Direction::DL], "DL");
    }

    std::cout << "CSV outputs:\n"
              << "  " << prefix << "greenran_ue.csv\n"
              << "  " << prefix << "greenran_slice.csv\n"
              << "  " << prefix << "greenran_sla.csv\n\n";

    // ---- Cleanup IPC ----
    if (g_ipcEnabled)
    {
        if (g_semMetricsReady)
        {
            sem_close(g_semMetricsReady);
            std::string name = "/rslaq_metrics_" + g_simUuid;
            sem_unlink(name.c_str());
        }
        if (g_semControl)
        {
            sem_close(g_semControl);
            std::string name = "/rslaq_control_" + g_simUuid;
            sem_unlink(name.c_str());
        }
        std::cout << "[GreenRAN] IPC semaphores cleaned up.\n";
    }

    Simulator::Destroy();
    return 0;
}
