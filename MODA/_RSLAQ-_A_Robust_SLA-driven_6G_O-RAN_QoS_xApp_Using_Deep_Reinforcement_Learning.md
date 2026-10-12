JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 1
RSLAQ - A Robust SLA-driven 6G O-RAN QoS
xApp using deep reinforcement learning
Noe M. Yungaicela-Naula, Vishal Sharma, Sandra Scott-Hayward
Abstract—The evolution of 6G envisions a wide range of
applications and services characterized by highly differentiated
```
and stringent Quality of Service (QoS) requirements. Open
```
```
Radio Access Network (O-RAN) technology has emerged as a
```
transformative approach that enables intelligent software-defined
management of the RAN. A cornerstone of O-RAN is the RAN
```
Intelligent Controller (RIC), which facilitates the deployment of
```
```
intelligent applications (xApps and rApps) near the radio unit. In
```
this context, QoS management through O-RAN has been explored
```
using network slice and machine learning (ML) techniques.
```
Although prior studies have demonstrated the ability to optimize
RAN resource allocation and prioritize slices effectively, they
have not considered the critical integration of Service Level
```
Agreements (SLAs) into the ML learning process. This omission
```
can lead to suboptimal resource utilization and, in many cases,
service outages when the target Key Performance Indicators
```
(KPIs) are not met. This work introduces RSLAQ, an innovative
```
xApp designed to ensure robust QoS management for RAN
slicing while incorporating SLAs directly into its operational
framework. RSLAQ translates operator policies into actionable
configurations, guiding resource distribution and scheduling for
```
RAN slices. Using deep reinforcement learning (DRL), RSLAQ
```
dynamically monitors RAN performance metrics and computes
optimal actions, embedding SLA constraints to mitigate conflicts
and prevent outages. Extensive system-level simulations validate
the efficacy of the proposed solution, demonstrating its ability
to optimize resource allocation, improve SLA adherence, and
```
maintain operational reliability (> 95%) in challenging scenarios.
```
Index Terms—O-RAN, QoS, Machine Learning, Network Slic-
ing, xApp, SLAs
I. INTRODUCTION
5
G/6G introduces services with stringent Quality of Service
```
(QoS) requirements, such as enhanced mobile broadband
```
```
(eMBB), ultra-reliable low-latency communications (URLLC),
```
```
and machine-type communications (MTC). Operating these
```
services is highly challenging, as real deployments can contain
multiple instances of these services with diverse QoS require-
```
ments. In the radio access network (RAN), this challenge
```
scales with dynamic user behavior, limited radio resources,
and fast channel condition variations.
RAN slicing has been considered one of the most promising
techniques to manage the QoS in 5G/6G [1]. In a sliced RAN,
```
Service Level Agreements (SLAs) are used between the net-
```
work operator and the tenants to declare expected performance
and quality of the services details [2]. Requirements of the
service instances are specified in terms of key performance
```
indicators (KPIs), such as throughput and latency. Violation of
```
The authors are with the Centre for Secure Information Technologies
```
(CSIT), Queen’s University Belfast, Belfast, Northern Ireland, UK.
```
E-mail: n.yungaicela@qub.ac.uk
SLAs can have severe consequences for the network operator,
such as financial penalties, reputational damage, and more
```
critically, legal and regulatory implications (e.g., in sectors
```
```
involving critical services like healthcare).
```
To maximize their revenues and avoid SLA violations,
network operators must allocate RAN resources efficiently,
```
dynamically, and in an automated manner. Open RAN (O-
```
```
RAN) offers a solution enabling dynamic resource manage-
```
```
ment through the RAN Intelligent Controller (RIC). O-RAN
```
helps translate the high-level requirements expressed by the
```
operator in SLAs into low-level network parameters (KPIs).
```
```
Furthermore, an intelligent application (xApp) within the RIC
```
can help keep the KPIs of different slices near the target KPIs
```
(SLAs). The xApp can utilize machine learning (ML) and real-
```
time metrics from the RAN and user behavior to optimize
resource allocation per slice. The xApp must guarantee that
the slices operate independently, i.e., the performance in
one slice does not negatively influence the performance of
other slices. Furthermore, the slices can have competing or
conflicting QoS requirements. Lastly, robustness is required
in SLA management to guarantee that the SLAs are reliably
maintained in different scenarios, e.g., varying traffic.
In practice, SLA management with O-RAN presents a real
challenge [3]. Therefore, the latest approaches to RAN slicing
```
have leveraged deep reinforcement learning (DRL) to optimize
```
resources while meeting QoS requirements. However, these
approaches focus on resource optimization, such as providing
high throughput for eMBB users and low latency for URLLC
users. While some of these approaches address QoS conflicts
through prioritization, they often fail to fully model the broader
network operator intents expressed in SLAs, which require
finer granularity. This limitation results in two primary issues:
```
(1) suboptimal resource utilization; and (2) reduced reliability
```
in QoS compliance. For example, if the throughput for an
eMBB slice is being maximized and exceeds the agreed SLA,
the surplus resources could be reallocated to other slices
that are falling short of their SLAs or released to enhance
energy efficiency in the system. Moreover, if reliability is not
considered, using only priorities to decide the distribution of
resources for QoS operations can lead to resource starvation
```
in slices with lower priority, potentially causing outages (KPIs
```
```
fall below the agreed thresholds). In 6G, where most slice
```
types demand high reliability, such outages are unacceptable.
By modeling the distinct SLA requirements of the slices
in a more granular manner and incorporating them into the
DRL learning process, a better balance of resource allocation
can be achieved, optimizing resources while also meeting the
```
target key performance indicators (KPIs). In this work, we
```
This article has been accepted for publication in IEEE Transactions on Mobile Computing. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TMC.2026.3666787
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: UNIVERSIDADE FEDERAL DO PARA. Downloaded on February 26,2026 at 18:13:50 UTC from IEEE Xplore. Restrictions apply.
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 2
propose RSLAQ, an xApp that reliably controls the QoS of
different services in the RAN. The xApp receives policies
from the operator through the RIC as target KPIs and uses
a DRL agent to guarantee that these targets are met while
optimizing the resource share between different services. First,
a general model is presented, followed by an evaluation of
a specific design for operating three slices: eMBB, URLLC,
and MTC. This work builds on our previous paper [4] and
includes substantial extensions in methodology, robustness,
experiments, and discussion.
The contributions of this work are as follows:
• We provide insights into the existing strategies for operat-
ing the QoS in 3GPP 5G NG-RAN and O-RAN through
a comprehensive review of recent studies from academia
and industry.
• We present RSLAQ, a DRL-based xApp for the reliable
operation of QoS RAN that adheres to 3GPP and O-
RAN specifications. This design incorporates granular
SLA requirements as target KPIs, integrating them into
the learning process of the DRL agent.
• We demonstrate the effectiveness of our solution by
using highly-detailed system-level simulations that follow
3GPP specifications.
The rest of this document is organized as follows: Section II
presents the background on QoS operations within 3GPP and
O-RAN and the existing ML-based strategies. The architecture
of the system proposed in this study is detailed in Section III.
Section IV presents RSLAQ, our QoS xApp that includes the
design of the DRL approach. Experimental results are reported
in Section V. Discussion and conclusions are provided in
Sections VI and VII, respectively.
II. BACKGROUND AND STATE OF THE ART
This section presents a comprehensive review of QoS oper-
ation in 3GPP 5G and O-RAN.
A. 3GPP QoS framework
In the 3GPP 5G framework, the Access and Mobility
```
Management Function (AMF) and the User Plane Function
```
```
(UPF) (which are in the 5GC) filter the traffic from different
```
```
services and map them to QoS flows (using QoS identifiers,
```
```
QFIs) based on agreed service requirements. Within the RAN,
```
```
the Service Data Adaptation Protocol (SDAP) associates QFIs
```
```
with data radio bearers (DRBs) managed by the RLC/MAC
```
protocols [5]. The MAC layer then schedules resources using
```
queue management algorithms such as Round Robin (RR),
```
```
Proportional Fairness (PF), or Best Quality Channel Indicator
```
```
(BCQI) [6].
```
The QoS strategy used in the 3GPP 5G QoS framework
is semi-dynamic because the DRBs are set up in a pre-
determined and fixed manner. Furthermore, previous studies
have highlighted a limitation regarding flow granularity [7].
Despite the 5GC supporting relatively high granularity with
up to 64 QFIs per UE, the 5G NR restricts the number of
possible DRBs per UE to 32. This restriction can complicate
the prioritization of similar flows, potentially causing data
starvation for some flows. Finally, in the 3GPP 5G QoS
SMO
Non-RT RIC
CU/DU/RU
Near-RT RIC
O1
```
(1) Initial setup
```
```
(default response)
```
```
(4) Updating KPI targets (policy)A1
```
rApp
xApp
```
(9) Long-term
```
monitoring
```
(8) Short-term
```
monitoring
```
(3) SLA to KPI targets
```
```
(5) Per slice/UE/flow
```
resource assignment
```
(7) Resource
```
allocation
```
(6) Control commands
```
```
(2) SLA
```
E2 Node E2 Agent
KPM SMRC SM
Network operator
Scheduler ~1ms
xApp 10ms - 1s
rApp > 1 s
Fig. 1: QoS operation in O-RAN. The numbered steps ex-
emplify the operation of QoS with network slicing and SLA
assurance proposed by the O-RAN Alliance [2], [8], [10]–[12].
framework, application requirements and channel variations
are not closely integrated with RAN resource optimization
mechanisms. Typical MAC schedulers utilize physical and
logical channel metrics, but are generally unaware of QoS
requirements. For example, PF aims to maximize cell through-
```
put by considering channel quality indicators (CQI) while
```
ensuring a minimum level of service for all users. RR scheduler
distributes resources equally among users. BCQI prioritizes
users with the highest CQI for transmission, maximizing cell
throughput. Variations of these algorithms, such as weighted
proportional fairness, can prioritize services during scheduling.
However, they are only capable of reacting to short-term
changes, rely on local information, and do not account for all
intentions of the network operator. These limitations of 3GPP
are being addressed by O-RAN technology, as detailed next.
B. O-RAN assisted QoS
```
In O-RAN, the RAN is split into the central unit (CU),
```
```
the distributed unit (DU), and the radio unit (RU). These
```
```
components expose E2 service models (E2SMs) to the RIC.
```
The RIC uses both the Non-RT RIC and the Near-RT RIC
to manage RAN resources with millisecond and second-level
granularity, respectively [8], [9].
The operation of RAN slicing for QoS and SLA assurance
proposed by the O-RAN Alliance is depicted in Fig. 1 [2], [8],
[10]–[12]. The SMO uses the O1 interface to provide the initial
```
and default configuration of the slices (1), i.e., the proportion
```
of resources allocated for a newly created slice, and the MAC
scheduler that distributes the resources for each UE within
the new slice. Next, the Non-RT RIC captures the SLAs from
```
the SMO (2). The Non-RT RIC is a subcomponent of the
```
```
Service Management and Orchestration (SMO), thus it has
```
```
access to the data that is captured from the RAN (through the
```
```
O1 interface; see (9) in Fig. 1). Using SLAs and the long-term
```
performance metrics of the RAN, the Non-RT RIC is able to
```
translate the SLAs to target KPIs (3). The latency and bit
```
error rate are examples of target KPIs estimated by the Non-
RT RIC to fulfill the SLA requirements of URLLC services.
This article has been accepted for publication in IEEE Transactions on Mobile Computing. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TMC.2026.3666787
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: UNIVERSIDADE FEDERAL DO PARA. Downloaded on February 26,2026 at 18:13:50 UTC from IEEE Xplore. Restrictions apply.
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 3
TABLE I: Existing works for QoS in O-RAN developed
by industry and research bodies. ROB = Robustness analysis
under nominal and stress scenarios.
```
Work Approach DRL SLA ROB(1)
```
Policy
level
```
(2)
```
Resources
per-slice
```
(3)
```
Per-slice
scheduling
2021 [13] - - ✓ ✓ - -
2022 [14] - - ✓ ✓ - -
2022 [15] - - ✓ - - -
2022 [16] - - ✓ - - -
2022 [17] ✓ ✓ ✓ - -
2022 [18] ✓ ✓ ✓ - ✓ -
2022 [19] ✓ ✓ ✓ - ✓ -
2022 [20] ✓ - - - - -
2023 [21] - ✓ - - - -
2023 [22] - ✓ ✓ ✓ - -
2023 [23] ✓ - - - - -
2023 [24] - ✓ - ✓ ✓ -
2023 [25] - ✓ - ✓ ✓ ✓
2024 [26] ✓ ✓ - ✓ ✓ -
2024 [27] ✓ ✓ - ✓ ✓ -
2024 [28] ✓ ✓ ✓ ✓ -
2024 [29] ✓ - - - - -
2025 [30] - ✓ - ✓ - -
2025 [31] ✓ ✓ - ✓ ✓ -
2025 [32] - ✓ - ✓ - -
RSLAQ ✓ ✓ ✓ ✓ ✓ ✓
The Non-RT RIC can use an rApp for the prediction of events
```
(such as traffic congestion) and optimized generation of target
```
KPIs for different slices. The target KPIs are sent to the Near-
```
RT RIC as A1 policies (4). The Near-RT RIC enforces these
```
```
KPIs in the RAN (5). To achieve this, the Near-RT RIC uses
```
an ML-based xApp that optimizes the assignment of resources
to different slices according to the target KPIs and the RAN
```
resource status (8). Finally, RAN control is enabled for the
```
```
RAN slice operation through the E2 service models (7).
```
Currently, there are some efforts toward achieving the
QoS framework presented in Fig. 1. Table I shows existing
studies that use O-RAN for QoS in 6G RAN. There are
```
three approaches: (1) Policy-level management; (2) Resource
```
```
allocation per RAN slice; and (3) Per-slice scheduling.
```
```
1) Policy-level management: Early approaches considered
```
how to improve some of the mechanisms from the 3GPP QoS
framework by using application domain information, user in-
puts, and edge services. The SDAP protocol that maps QFIs to
DRBs has been first targeted in the design of applications that
provide RAN control of UE-specific DRB-level QoS. In this
context, an xApp automates and optimizes the mapping of the
QFI flows into the DRBs. For example, [23] and [29] present
xApps that change the priority of the QoS flows of specific
```
user equipment (UEs) according to the policies provided by
```
the operator. In an emergency scenario, if two applications are
running, e.g., video monitoring and file transfer streams, the
operator elevates the priority of specific cameras by sending
```
policies to the RIC by using the A1 interface (see (4) in Fig.
```
```
1). To support these approaches, the estimation or prediction of
```
```
the QoS parameters, such as network-level (congestion), cell-
```
```
level (throughput, delay, packet error, packet loss), and UE-
```
```
level (throughput and latency performance), is of paramount
```
importance [29] [33]. In this context, the SMO collects long-
```
term statistics from the RAN through the O1 interface (see
```
```
(9) in Fig. 1), while the Non-RT RIC may employ ML-based
```
```
applications (rApps) to make these predictions.
```
Optimization mechanisms to enhance traffic mapping be-
tween various flows have also been investigated using O-
RAN features. The work in [20] proposed an optimization
```
mechanism (mixed integer programming) that maps QoS flows
```
into multiple routes by considering the retransmission of the
packets. A single CU is connected to a pool of DUs. The DU
pool contains a set of processors for packet retransmission
scheduling. Under this approach, the delay time for URLLC
```
traffic and the resource utilization (processors) for eMBB traf-
```
fic are minimized. Similarly, the authors in [34] experimented
with a traffic prioritization mechanism to manage live video
```
streaming (MBB) and remote car control (LLC). The mapping
```
algorithm considers the state of each queue and the weighting
of the allocation of radio resources according to the prioritized
traffic volume.
Note that the approaches discussed in this section operate
within the application and data planes, utilizing protocols of
the 3GPP NR QoS framework, such as DRB formation. The
following approaches represent efforts aimed at leveraging ML
to optimize radio resource scheduling in the RAN according
to the requirements of different applications and the network
state.
```
2) Resources per-slice: In this approach, an xApp is de-
```
signed to optimize the distribution of the physical resource
```
blocks (PRBs) per slice. For example, the work in [35] pro-
```
posed the design of a DRL-based agent that allocates PRBs for
three defined slices: eMBB, URLLC, and MTC. This approach
maximizes the traffic rate for eMBB and the number of PRBs
for MTC, and minimizes the buffer size for URLLC. Later, this
work was extended by the authors in [22]. They reconfigured
the reward function of the DRL by providing weights for each
slice in order to avoid collisions between competing target
KPIs. The work in [32] experimented with the same approach
in an open-source O-RAN testbed, emphasizing weighted
traffic prioritization for URLLC, eMBB, and MTC.
```
3) Per-slice scheduling: In this approach, the xApp controls
```
how the PRBs are distributed within each slice. A common
technique is to find the optimal sequence of the MAC-
```
scheduling algorithms (e.g., RR and PF) to apply in each slice
```
based on network conditions. For example, in [35] and [13],
```
the authors established 3 slices (eMBB, URLLC, and MTC)
```
and three DRL agents, one per slice, that dynamically select
the best scheduler at a given time. As reported in these works,
this dynamic selection significantly outperforms the individual
operations of the schedulers.
Another approach has been to replace the conventional
MAC schedulers with optimized mechanisms. For example,
the work in [14] proposed a DRL method that multiplexes
```
the spectrum resources (PRBs) to optimize the cumulative
```
throughput of eMBB and URLLC users. The mitigation
```
of inter-numerology interference (INI) is considered in this
```
This article has been accepted for publication in IEEE Transactions on Mobile Computing. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TMC.2026.3666787
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: UNIVERSIDADE FEDERAL DO PARA. Downloaded on February 26,2026 at 18:13:50 UTC from IEEE Xplore. Restrictions apply.
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 4
method. The work in [15] studies scheduling policies for PRB
allocation, including common mechanisms, such as MaxCQI,
and other customized strategies. Their analysis also introduced
centralized and distributed controllers and handover opera-
tions to analyze the effects on QoS. Although the distributed
approach presents higher overhead and complexity in the
control plane, the QoS is maintained compared to the single-
controller approach. In a similar study in [16] evaluated QoS-
```
unaware and -aware (e.g., exponential/proportional fairness)
```
MAC scheduling policies within three specific slices: VoIP
```
(LLC), Video (eMBB), and CRB (mTC). As expected, delay-
```
aware mechanisms outperform unaware methods in all sce-
narios. Note that [14]–[16] assume direct assignment of PRBs
from the xApp, which may be impractical due to the response
```
time of xApps (≥ 10 ms).
```
All these previous works represent partial efforts to achieve
the QoS operation presented in Fig. 1. Recent works have at-
tempted to merge different strategies to achieve complete QoS
management. For example, the authors in [22], [35] reported
that running a single DRL agent to control the scheduling
```
policy for slices and multiple DRL agents (one per slice)
```
for spectrum resource scheduling in each slice significantly
increases the performance with respect to the 5G NR base
line schedulers. On the other hand, Rimedo Labs and the
```
Open Networking Foundation (ONF) created the QoS-based
```
```
Resource Allocator (QRA) xApp, which follows A1 rules
```
[21]. These policies include equal distribution, reservation,
and preference. The xApp allocates resources to various slices
```
based on these rules and RAN parameters (e.g., SNR and
```
```
throughput). However, the application of ML for resource
```
distribution within each slice was not explored.
```
4) SLA-driven QoS in O-RAN: In a recent demonstration,
```
[18] highlighted the importance of ensuring SLAs for UEs in
specific slices. The policies are received from the operator
through the A1 interface. However, their work focused on
a single type of slice aimed at guaranteeing the minimum
throughput per UE. Our study extends this analysis to systems
where multiple types of slices coexist, handling challenges
such as conflicts between SLA requirements of slices and
priority management. Furthermore, [18] did not specify the
use of ML. By incorporating intelligent methods such as DRL,
we can better model system behavior and optimize resources,
even under varying conditions. A similar demonstration was
conducted in [19], focusing on a single type of slice but
analyzing different latency requirements. To guarantee the
target delay for UEs, they reserved PRBs per slice and adjusted
the UE/slice priority levels and packet delay budget. However,
the use of ML was not specified in this work.
There are a few works that studied SLA with DRL. [28]
showed that dynamic SLA requirements can be addressed
using DRL. Although this work offered valuable insights for
our approach, it focused on just one type of slice, which is
a more simplified scenario compared to our method. Addi-
tionally, their study only considers per-slice SLAs and does
not address per-UE KPIs, which limits the granularity needed
for 5G/6G applications. [17] introduced an evolutionary DRL
approach to address QoS for three slices: eMBB, MTC, and
URLLC. While the approach was demonstrated in terms of
maximizing QoS metrics, such as average data rate for eMBB,
capacity for MTC, and delay for URLLC, it did not analyze
SLA compliance. Furthermore, the approach involves directly
```
assigning PRBs (overriding MAC scheduling) from the xApp,
```
which might be impractical due to the response time of
```
the xApp (≥ 10 ms), as previously noted. The authors in
```
[30] presented a DRL-based solution for inter-slice resource
```
allocation. The agent uses slice-level KPIs (queue length and
```
```
head of buffer) to maximize data transmission and minimize
```
delay. They tested the solution using system-level simulations
and an emulated testbed. Unlike our solution, their DRL does
not reflect SLA compliance or the network operator policy.
In a more recent approach, [25] addressed dynamic resource
distribution by combining a regression-based network model,
a risk-constrained DRL agent, and a traffic prediction module.
Their approach learns the complex relationship between QoS
and resource allocation. While tested under varying traffic
```
conditions, it focuses on a single slice type (eMBB) and
```
does not resolve conflicts among heterogeneous slices, limiting
applicability in realistic 5G/6G scenarios. [24] proposed a
Federated DRL architecture for scalable and distributed 6G
RAN slicing orchestration. The approach minimizes SLA
violations by deploying local DRL agents at the edge, grouped
into specialized clusters based on similar slice traffic demands.
However, implementation, integration, and maintenance com-
```
plexity for managing distributed AI agents (xApps) must be
```
considered. Additionally, reliability analysis was not presented
in this study. [27] addressed SLA-aware slicing using the
```
Soft Actor-Critic (SAC) algorithm, considering diverse SLA
```
types and slice configurations. They employed slice-level
priorities and demonstrated superior performance over baseline
schedulers. The authors focused on nominal conditions without
outage analysis or stress scenarios. In contrast, beyond opti-
mization, RSLAQ emphasizes robustness, optimization, and
deployability. In addition, we incorporate policy-free slices,
intra-slice scheduling, and soft SLAs. Moreover, [27] operates
at 1 ms TTI, which may not be practical for typical xApp
response times. SafeSlice [31] is a DRL-based slicing solu-
tion that introduces safety guarantees for SLA compliance,
particularly strict latency requirements. It combines a risk-
sensitive reward function with an external supervised learning
safety layer to project unsafe actions into safe ones, reducing
latency violations. In contrast, RSLAQ achieves robustness
by design without external ML layers, minimizing complexity
and avoiding black-box constraints that may hinder exploration
of high-reward states. Additionally, RSLAQ supports diverse
slice types, reflecting realistic 5G/6G deployment scenarios.
[26] proposed a constrained multi-agent RL approach for
flexible RAN slicing that adapts to variable slice numbers.
Their solution assigns dedicated agents per tenant and uses
sequential decision-making, focusing on large-scale slice vari-
ability. In contrast, RSLAQ emphasizes robustness by design
and outage-aware optimization under stress scenarios.
Table I summarizes the focus and contributions of the
previous work. RSLAQ offers a simple yet effective SLA
management solution with robustness by design, avoiding ex-
ternal ML layers that add complexity and reduce determinism.
RSLAQ uses a single agent to learn inter- and intra-slice
This article has been accepted for publication in IEEE Transactions on Mobile Computing. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TMC.2026.3666787
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: UNIVERSIDADE FEDERAL DO PARA. Downloaded on February 26,2026 at 18:13:50 UTC from IEEE Xplore. Restrictions apply.
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 5
Operator RSLAQ CU/DU/RU
2. Send policy:
```
{Slice ID: (SLA, Priority)}
```
8. Report statistics
Non-RT RIC
1. Provide policy
for RAN slicing
4. Send target KPI:
```
{Slice ID: (target KPI, weight)}
```
Near-RT RICSMO
3. Translation from policy
to target KPI
6. Send target parameters
```
{Slice ID: (rsh, sch)}
```
5. Calculate the resource share
and scheduler
7. Monitor RAN status
and assign resources
and scheduler to slice
```
Loop: RAN SlicingLoop: RAN Slicing
```
```
Loop: MAC schedulerLoop: MAC scheduler
```
Fig. 2: Intent-based SLA management. The high-level policies of the network operator are translated and enforced into the
RAN using the proposed framework.
```
actions, supports diverse slice types (including policy-free
```
```
and priority-based slices), and incorporates soft SLAs. We
```
validate under realistic 5G/6G stress scenarios with outage-
```
aware optimization and O-RAN-compliant timing (10 ms),
```
ensuring deployability.
III. SYSTEM ARCHITECTURE
This section presents the proposed QoS framework for
reliable SLA compliance, encompassing the workflow from
operator intent to resource allocation across and within net-
work slices.
A. Intent-based SLA management
Fig. 2 illustrates the operational workflow of the intent-
based SLA management system. The network operator pro-
vides the parameters for the definition of the slices. Three
types of slices are considered in O-RAN: Reserved, Policy,
and No-Policy slices [21], [36]. A Reserved slice receives a
static quota of resources. Policy and No-Policy slices share the
remaining resources, where the former adhere to SLAs and the
latter are not bound by any policies. Although the design of
RSLAQ provides flexibility to include the three types of slices,
this study covers only Policy and No-Policy slices. DRL is used
to optimally distribute resources between these slices.
In Fig. 2, the slice specifications for Policy and No-Policy
slices are sent to the Non-RT RIC. In terms of Policy slices, a
set of slices is configured with different SLAs and priorities:
the lower the value, the higher the priority. For example,
for URLLC, latency requirements are usually considered,
whereas for eMBB, the achieved throughput is taken into
account. A critical parameter in the slice configuration is the
maximum number of UEs per slice, as this constraint enables
consistent adherence to specified SLA requirements. In the
URLLC example, if the number of users connected to this
slice is very high, the system may not guarantee the latency
requirements given its resource constraints. In the context
of 6G applications, reliability in guaranteeing SLAs is also
important and is defined as the percentage of times the SLA
requirements have been met. Note that any conflicts between
SLAs must be resolved in the Non-RT RIC before enforcing
them in the system. The intent-level conflicts lie beyond the
scope of this paper.
The high-level specifications of SLA and priorities defined
for the slices are translated by the Non-RT RIC to weights
```
and target KPIs (see step 3 in Fig. 2). Weights are calculated
```
from the priorities and SLAs are translated into target KPIs
that define the outage probability limits. These calculations are
explained in detail in Section IV.
```
In the following step (step 4 in Fig. 2), the Non-RT RIC
```
This article has been accepted for publication in IEEE Transactions on Mobile Computing. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TMC.2026.3666787
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: UNIVERSIDADE FEDERAL DO PARA. Downloaded on February 26,2026 at 18:13:50 UTC from IEEE Xplore. Restrictions apply.
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 6
```
{
```
" network_slices ": [
```
{
```
" slice_name ": "URLLC",
" weight ": 0.40,
```
" target_kpis ": {
```
```
" outage_kpis ": {
```
" k_out_1 ": "latency per UE > 5ms",
" k_out_2 ": "packet_loss_rate per
UE > 0.1%",
" k_out_3 ": "bandwidth_mbps per
slice < 10mbps",
" reliability_percent ": "99.999%"
```
},
```
```
" soft_kpis ": {}
```
```
}
```
```
},
```
```
{
```
" slice_name ": "eMBB",
" weight ": 0.37,
```
" target_kpis ": {
```
```
" outage_kpis ": {
```
" k_out_1 ": "packet_loss_rate per
UE > 1%",
" k_out_2 ": "bandwidth_mbps per
slice < 1000mbps",
" reliability_percent ": "99.99%"
```
},
```
```
" soft_kpis ": {
```
" k_soft_1 ": "bandwidth_mbps per
slice > 1500mbps"
```
}
```
```
}
```
```
},
```
```
{
```
" slice_name ": "MTC",
" weight ": 0.23
```
}
```
]
```
}
```
Fig. 3: Example of A1 policy.
sends target KPIs and weights to RSLAQ, which is located
in the Near-RT RIC, in the form of an A1 policy. Fig. 3
shows an example of this A1 policy where the weights of
the slices are assigned a proportional value to their provided
priorities. Note that the sum of the weights must be 1. In
```
addition, outage kpis and soft kpis (see definitions in Section
```
```
IV) are the main elements that are used by RSLAQ to estimate
```
the resource distribution to be sent to the RAN components.
RSLAQ is designed to use DRL to calculate the resource
```
share (rsh) and RAN scheduler (sch). To achieve this, it
```
continuously monitors the statistics of the RAN and sends
```
control commands to the RAN every 10 ms (5G framework
```
```
duration). The intents provided by the network operator are
```
translated into numeric inputs at this layer. The two main con-
trol parameters sent to the RAN component are the distribution
```
of resources (rsh) and the scheduler to be used (sch).
```
The MAC scheduler is responsible for enforcing the rules
```
generated by RSLAQ (step 7 in Fig. 2). Notably, the scheduler
```
operates within its own control loop, which functions at 1 ms
intervals. The scheduler has the capability to override the
control inputs of RSLAQ based on RAN metrics. For instance,
if RSLAQ assigns a significant amount of transmission re-
sources to an eMBB UE but the scheduler identifies that the
user’s transmission buffer is empty, the scheduler disregards
the RSLAQ command. Instead, it redistributes the resources
among other users while maintaining the proportionality dic-
tated by RSLAQ.
B. RSLAQ design
The core element of the proposed framework is the xApp
that works as follows: First, RSLAQ receives the target KPIs
```
(SLA) from the Non-RT RIC through the A1 interface. Then,
```
```
the xApp monitors the E2 Node (DU) to gather statistics.
```
For the case study analyzed in this document, statistics of
```
throughput (thr), transmitted bytes (btx), buffer status (bfs),
```
```
and resource share (rsh) per UE are monitored. In this study,
```
a single cell is considered. RSLAQ uses a DRL agent to select
```
actions of resource proportion distribution (pj ) and scheduling
```
```
selection (sch). Finally, RSLAQ reports whether the target
```
KPIs are being met and provides information on any failures,
such as lack of resources. The design of RSLAQ incorporates
features that ensure robustness against variable user traffic
patterns, manage resource conflicts during congestion events,
minimize service outages, optimize resource utilization, and
maintain slice isolation. A detailed architectural analysis is
presented in Section IV.
Note that the slice-aware scheduler is required on the E2
node. Furthermore, KPM and RC SMs are needed to enable
monitoring and control of the E2 node. In the proposed design,
the DRL agent captures metrics and executes actions every
10 ms.
C. Slice-Aware scheduler
The proposed framework is designed for Time Division Du-
```
plex (TDD). TDD operates with two independent schedulers
```
```
for downlink (DL) and uplink (UL). The focus in this study is
```
on the DL, which can be extended to UL. Furthermore, while
5G offers high flexibility in resource scheduling in frequency
and time domains, our design aims to reduce computation
and energy consumption by following standard practice and
```
focusing on resource optimization at the frame level (defined
```
```
as 10 ms in this context).
```
The slice-aware scheduler works as follows: First, the TDD
pattern is configured by the 3GPP 5G NR radio resource
```
control (RRC) protocol. Subsequently, in the TDD pattern,
```
```
slots (1 ms each) designated for UL or DL are independently
```
managed by the xApp commands. In the DL slots, for ex-
ample, the proportion of DRBs is distributed among the slices
according to the xApp command. Finally, based on the selected
```
scheduler (e.g., RR), resources within each slice per slot are
```
allocated to users. This distribution is probabilistic rather than
deterministic. For example, if three slices are considered and
the distribution probabilities are [0.6, 0.2, 0.2], it means each
time a PRB is considered by the scheduler, there is a 0.6
probability it will go to the first slice and a 0.2 probability
it will go to the second or third slice. This probabilistic
```
distribution is applied at the full frame level (10 ms).
```
This article has been accepted for publication in IEEE Transactions on Mobile Computing. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TMC.2026.3666787
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: UNIVERSIDADE FEDERAL DO PARA. Downloaded on February 26,2026 at 18:13:50 UTC from IEEE Xplore. Restrictions apply.
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 7
In terms of the MAC layer, the Hybrid Automatic Repeat
```
Request (HARQ) protocol is activated to ensure reliable trans-
```
mission over the air interface through error correction. Further-
more, the MAC scheduler prioritizes UEs with retransmissions
```
(HARQ). However, only one retransmission is allowed per
```
```
UE per Transmission Time Interval (TTI). Additionally, the
```
```
scheduler allocates resources to UEs only if (i) there are
```
```
remaining resources after handling retransmissions; (ii) the
```
```
UEs have not received resources for retransmissions; (iii)
```
```
the UEs have free HARQ processes; and (iv) the UEs have
```
data ready to transmit. These conditions make the task of
the DRL agent challenging, as the resulting states of actions
are not deterministic. Nevertheless, reinforcement learning
can effectively address non-deterministic scenarios, which is
demonstrated in our design.
IV. DESIGN OF DRL AGENT
This section introduces the DRL agent designed to optimize
resources and ensure SLA compliance, thereby, delivering QoS
for demanding 6G applications.
A. DRL formulation
The design of the DRL agent is intended to be as simple
as possible due to the response time requirements. Three
important definitions are needed in this design: state, actions,
and reward function. These definitions are presented below.
```
1) State: Let M be a set of slices such that M =
```
```
{m1, m2, mj , . . . , mJ }. Also, Λj is the set of UEs assigned
```
to slice mj . The state of the network is defined based on the
statistics gathered from the E2 node as follows:
```
s =
```



btxm1 . . . btxmJ btxcell
bfsm1 . . . bfsmJ bfscell
rshm1 . . . rshmJ rshcell
tdpm1 . . . tdpmJ tdpcell


```
 , (1)
```
where btx, bfs, rsh, tdp are the bytes to transmit, the buffer
```
status, the share of resources, and (transmitted) dropped bytes,
```
respectively. The last column represents the total parameters
```
per cell, including all existing slices (Policy and No-Policy
```
```
in this study), and is structured to support Reserved slices in
```
```
future work. The parameters in (1) were selected to reflect the
```
dynamics of the most common types of slices: btx and bfs help
track eMBB and MTC performance, while bfs and tdp assist
```
in monitoring URLLC performance. s ∈ R4×(J+1), where J
```
is the number of slices.
```
2) Actions: The DRL agent can control the proportion of
```
PRBs per slice pj and the type of scheduler used inside the
```
slices (sch). Note that in general pj = {z : z ∈ [0,1] and z ∈
```
```
R}. Using this definition would result in a DRL agent with
```
an infinite number of actions. To prioritize the speed of DRL
```
learning and avoid unreasonable selections of pj (e.g., P pj >
```
```
1), we discretize and constrain pj as follows: Consider J slices
```
in the system. Let pj ∈ P denote the proportion of PRBs for
the jth slice. Therefore,
```
P = [x|x ∈ {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1}].
```
```
(2)
```
To introduce slice isolation in RSLAQ, we take a semi-
dynamic approach to the distribution of radio resources in the
RAN. We balance between the static and dynamic resource
allocation strategies. In the static approach, a fixed number of
resources are allocated to each slice, ensuring strong isolation
but often leading to resource underutilization. Conversely,
the dynamic approach dynamically distributes all resources
without a pre-defined slicing strategy, enabling high resource
optimization and flexibility but risking resource starvation due
to the lack of isolation measures. RSLAQ operates as a hybrid
solution, allocating 50% of the resources statically while
distributing the remaining 50% dynamically in an optimized
manner. The static component psta is distributed according to
weights assigned by the network operator. The remaining 50%,
represented by popt, is dynamically optimized using DRL.
These components can be formally expressed as:
```
pstaj = ωj × 0.5 of total PRBs (3)
```
```
poptj = Θ(·) × 0.5 of total PRBs, (4)
```
```
where Θ(·) represents the DRL model that optimizes the
```
resources based on RAN statistics and SLA requirements. The
proportion of PRBs applied for each slice is
```
pj = pstaj + poptj. (5)
```
This approach achieves a balance between resource op-
timization and slice isolation. Table III shows an exam-
ple of slice isolation for three slices: eMBB, URLLC, and
MTC. pstaj takes values of [0.3333, 0.4000, 0.2667] × 50% =
[0.1667, 0.2000, 0.1333]. The minimum proportion of PRBs to
be assigned to each slice is that given by pstaj. Note that if the
slice is a no-policy slice, pstaj is 0. The maximum proportion
of resources to be provided to a slice is the total of PRBs
in the system minus the proportions that have been statically
assigned to the other slices. Finally, poptj can take a value
between 0 to 1, as it is multiplied by 50% corresponding to
the amount of resources assigned to the optimization part.
In terms of the scheduler, consider the scheduling options
for the slices as
```
SCH = {RR, PF, BCQI}. (6)
```
These three schedulers have been selected because they are
among the most well-known and widely used in resource dis-
tribution. This approach maintains compatibility with current
base station operations, requiring no modifications to their
native scheduling capabilities.
Finally, the set of actions A is as follows:
```
A = {({p1, . . . , pj , . . . , pJ }, sch) :
```
JX
```
j=1
```
```
pj = 1,
```
```
sch ∈ SCH, pj ∈ P ∀j). (7)
```
This article has been accepted for publication in IEEE Transactions on Mobile Computing. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TMC.2026.3666787
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: UNIVERSIDADE FEDERAL DO PARA. Downloaded on February 26,2026 at 18:13:50 UTC from IEEE Xplore. Restrictions apply.
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 8
```
TABLE II: SLA model for slices based on target KPIs (m1 = eMBB, m2 = URLLC, and m3 = MTC).
```
Condition Type Description Example
Optimization Independent/
competing
KPIs to optimize two or more slices that may be directly
or indirectly competing.
```
h11 = maximize throughput per UE for m1, h12 = minimize
```
buffer occupancy per UE for m2, h13 = maximize throughput
per UE for m3, ca minimize the cost of action.
SLA Outage The target KPI is critical to fulfilling the required SLA. k11 = minimum throughput for m1, k12 = maximum bufferoccupancy per UE for m2.
Soft Target KPIs can be missed without consequences for
SLA fulfillment, but they impact resource optimization.
```
k21 = maximum throughput per slice for m1.
```
TABLE III: Example of definition of actions for slice isolation.
.
```
Slice ωj (priority) Min pj = pstaj max pj poptj
```
```
eMBB 0.3333 (2) 0.1667 0.6667 (0,1)
```
```
URLLC 0.4000 (1) 0.2000 0.7 (0,1)
```
```
MTC 0.2667 (3) 0.1333 0.6333 (0,1)
```
```
3) Reward function for radio resource optimization: The
```
resources are optimized using the equation below.
```
ropt =
```
X
j
```
ωj hj + 1cost(α) , (8)
```
where hj represents the KPI to optimize for each slice j. ωj
allows to prioritize a slice if two or more slices have competing
KPI targets. Examples of optimization targets are provided in
```
Table II. cost(α) represents the cost associated with the action.
```
In this study, the cost of the scheduler is
```
cost(sch) =
```
```
(
```
1 if RR
```
2 otherwise, (9)
```
where the agent receives higher reward with RR because it
does not require CQI reports, making it less costly in terms
of frequency, computation, and energy compared to PF. RR
can be more effective in scenarios where users have similar
channel conditions.
We incorporate SLA requirements into DRL learning to
enable intent-based operations, ensuring optimal resource uti-
lization for the operator and QoS guarantees for users. In our
approach, we model SLAs as target KPIs and integrate them
into the reward function.
For each slice mj ∈ M , there is a set of target KPIs Kj =
```
{k1j , k2j , . . . , kij , . . . }, where kij represents the ith target KPI
```
```
for slice j. The SLA violation rate (vrslaj ) is
```
```
vrslaij = P (KPI < k
```
```
ij )
```
```
|Λj | . (10)
```
```
P (KPI < kij ) in Eqn. (10) denotes the normalized fre-
```
```
quency (empirical probability) in a timestep where the KPI
```
for slice j falls below its required threshold kij for all UEs
within that slice. This provides a generalized definition that
accommodates scenarios where multiple samples may exist
```
per timestep (e.g., statistics computed every 10 ms while the
```
```
agent acts every 100 ms). In our study, the agent acts every
```
TTI, so only one sample is considered per timestep. Finally,
dividing by the number of UEs in the slice, |Λj |, yields vrslaij ,
which is guaranteed to satisfy vrslaij ∈ [0, 1] and reflects SLA
violations across both sample and UE distributions.
Note that the requirements for some target KPIs may be
more critical than for others. In this context, Kout is the set
of target SLAs that are critical for fulfillment of the SLAs. On
the other hand, while KPIs in Ksof t do not impact SLA ful-
fillment, they are important for optimizing resource utilization.
For instance, failing to meet the maximum throughput limit per
```
slice for eMBB (see k21 in Table II) can lead to unnecessary
```
additional resource allocation, resulting in QoS that exceeds
the agreed SLA. While this is advantageous from the user
perspective, it is less favorable from the operator standpoint,
as the remaining resources could be better allocated to other
services.
The outage condition for a slice can be defined in terms
of the reliability parameter provided by the network operator.
```
Using (10), an outage condition for slice mj is defined as
```
φj =
```
(
```
1, if ∃ vrslaij such that vrslaij > 1 − reliabilityj
0, otherwise,
```
(11)
```
```
where φj ∈ {0, 1} is a binary value representing an outage
```
condition in slice mj . reliabilityj defines the minimum ac-
ceptable level that the operator considers an SLA violation
for slice j. For example, if the reliability for a specific SLA
is set to 99.999%, a deviation of 0.001% is tolerated before
an SLA violation, and thus an outage, is considered. In this
study, we assume reliability of 100%, so that any deviation
from the SLA is treated as an outage. Similarly, the condition
of soft SLA violations is defined by ρj , using kij ∈ Ksof t.
The reward function for SLA assurance is defined as follows:
```
r =
```



```
0 (Terminal s), if Pj ρj > 0
```
```
− Pj (φj ωj ) (Terminal s), if Pj φj > 0
```
ropt, otherwise,
```
(12)
```
where the first and second elements represent the outage
probability and soft probability, respectively, indicating the
failure to achieve any outage or soft target KPI. These cases are
```
terminal states (Terminal s, end of episode during learning).
```
```
ropt in Eqn. (12) is defined in Eqn. (8) and optimizes the
```
resources per slice and per UE within each slice.
B. Formulation for 3 slices: eMBB, URLLC, and MTC
The definitions of actions, states, and reward are provided
for the case study of 3 slices: eMBB, URLLC, and MTC.
```
The state in Eqn. (1) is given as s ∈ R4×4. The action set is
```
```
defined in Eqn. (7), where A = {a1, a2, . . . }. For three slices,
```
```
an example of this action is a1 = (p1, p2, p3, RR) where pj
```
This article has been accepted for publication in IEEE Transactions on Mobile Computing. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TMC.2026.3666787
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: UNIVERSIDADE FEDERAL DO PARA. Downloaded on February 26,2026 at 18:13:50 UTC from IEEE Xplore. Restrictions apply.
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 9
```
is given by (5). The number of possible actions is |A| = 198,
```
and thus A ∈ R198X4.
From the examples presented in Table II, the KPIs for
eMBB, URLLC, and MTC slices are grouped as follows: target
```
KPIs for eMBB are {k11 , k21 } and the optimization KPI is {h11};
```
```
the target KPI for URLLC is {k12 } and the optimization KPI
```
```
is {h12}; and the optimization KPI for MTC is {h13}.
```
The conditions in Kout define the outage probability as:
```
P (thr per slice for m1 < k11 ), (13)
```
```
P (bfs per UE for m2 > k12 ), (14)
```
and the conditions in Ksoft define the soft probabilities as:
```
P (maximum thr per slice for m1 > k21 ), (15)
```
and these probabilities are used to determine vrsla11, vrsla12,
```
and vrsla21 using Eqn. (10). These elements serve to calculate
```
```
φ1, φ2, ρ1 using Eqn. (11). Finally, these outage conditions
```
```
are integrated as the first (ρ1, soft) and second (φ1 and φ2,
```
```
outage) elements in Eqn. (12).
```
```
ropt in Eqn. (12), for this 3-slice example, is given as
```
```
h11 = ω11|Λ
```
1|
X
UE ∈Λ1
```
thr(UE), (16)
```
```
h12 = ω2e−(maxUE ∈Λ2 bfs(UE)), (17)
```
```
h13 = ω31|Λ
```
3|
X
UE ∈Λ3
```
thr(UE), (18)
```
```
cα = cost(sch). (19)
```
```
All components in Eqn. (16) - (19) are positive. Thus, if a
```
KPI requires minimization, such as for URLLC, it is converted
to a positive value through an exponential function. Addition-
```
ally, all elements must be normalized (e.g., we use maximum
```
```
achievable rates for the throughput normalization). Weights
```
ωj allow the definition of priorities per slice according to the
configurations of the network operator. In our experiments, we
consider priorities of [2 1 3] for eMBB, URLLC, and MTC
set by the operator, which translates to ωj = [0.33, 0.40, 0.27].
C. DRL algorithm
```
Dual Q-learning (DDQL) with experience replay is used
```
to solve the optimization problem. This solution is chosen
for its simplicity and timely response. Algorithm 1 presents
the DDQL implementation for RSLAQ. DDQL uses two deep
```
neural networks (DNNs) with the same structure: An Online
```
```
DNN with parameters θ (weights and biases) and a Target
```
```
DNN with parameters ˆθ. While the Online DNN (θ) is trained
```
in each learning step to decrease the loss function, the Target
```
DNN (ˆθ) is frozen to enhance learning stability. The Target
```
```
DNN (ˆθ) is periodically updated to match the Online DNN (θ)
```
after a pre-defined number of learning steps. The experience
replay mechanism stores past experiences in a dataset with
```
entries of the form ⟨s, α, r, s′⟩ (current state, action, reward,
```
```
next state). Furthermore, mini-batches of this dataset are used
```
to train the DNNs, which helps the agent reduce the number of
Algorithm 1 DDQL with experience replay
```
Input: ϵ, λϵ, ϵmin, γ, E, L, btsz, nsut, ntsr
```
```
Output: Best action ⟨pj , sch⟩ for each state
```
```
1: Initiate Online DNN (θ) and Target DNN (ˆθ)
```
2: Initiate replay memory V to capacity L
3: Set initial state s
4: for steps=1 to E do
5: Choose
α =
```
(
```
random if ϵ
```
maxαQ(s, α : θ) if 1 − ϵ
```
```
6: SET ACTION(α) ▷ Execute the action
```
```
7: [tstate, r] = GET REWARD(stats, Kout, Ksoft, reliability, α)
```
8: Observe s′
9: Store transition ⟨s, α, r, s′, tstate⟩ into V
10: if steps > btsz then
```
11: Retrain QNetwork(V , L)
```
```
12: ϵ = SET EPSILON DECAY(ϵ) ▷ Exponential epsilon decay
```
13: end if
```
14: if mod(steps, nsut) = 0 then
```
15: Update ˆθ of the Target NN with θ of the Online NN
16: end if
```
17: if mod(steps, ntsr) == 0 then
```
18: tstate = 1 ▷ Reset to initial state
19: end if
20: Set s = s′ ▷ Move to the new state
21: end for
22: Use final θ to retrieve the action α with the highest Q-value for each
state s
interactions that are required to learn and reduce the variance
in learning updates.
Algorithm 1 receives the exploration parameter ϵ, the explo-
ration decay rate λϵ, the minimum exploration decay rate ϵmin,
the discount factor γ, the number of learning steps E, the size
```
of the replay memory L, the minibatch size btsz (btsz <= L),
```
the number of steps before updating the Target DNN nsut,
and the number of steps before the environment is reset ntsr.
```
The output is the best action (pj and sch) for each state.
```
The algorithm first initializes the Online and Target DNNs
with θ and ˆθ, respectively, and the replay memory V to
```
capacity L (Lines 1-2). The agent goes through a finite number
```
of steps in a loop until it reaches a maximum of steps E
```
(Lines 4-21). In our design, a learning episode comprises a
```
sequence of steps that correspond to the states between an
initial state and a terminal state. A terminal state is defined
```
either through the reward function (outages; see Lines 27 and
```
```
30) or after every ntsr steps. Each step consists of selecting
```
and performing an action, changing the state, and receiving a
reward.
In Lines 5-20, the agent selects an action a ∈ A by using
```
the ϵ-greedy method (Line 5). Thereafter, the agent executes
```
```
the selected action a in the environment (Line 6), computes
```
```
the reward r, and gets the next state s′ (Line 8). Then the
```
agent stores the tuple ⟨s, a, r, s′, tstate⟩ in the replay memory
```
V (Line 9). The variable tstate is a binary indicator for
```
the terminal state, where 1 indicates a terminal state and 0
indicates a non-terminal state.
Once the replay memory contains more btsz experiences
```
(Line 10), the agent randomly takes a mini-batch (with size
```
```
btsz) from the replay memory V to train the Online DNN
```
```
(θ) (Line 11). In this process, for all samples in the mini-
```
```
batch, the agent predicts target by using the Online DNN (θ)
```
This article has been accepted for publication in IEEE Transactions on Mobile Computing. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TMC.2026.3666787
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: UNIVERSIDADE FEDERAL DO PARA. Downloaded on February 26,2026 at 18:13:50 UTC from IEEE Xplore. Restrictions apply.
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 10
```
and the Target DNN (ˆθ). After that, the agent uses gradient
```
descent and backpropagation algorithms to adjust the weights
and biases of the Online DNN, θ, and to minimize the Loss =
```
(target − Q(s, α))2.
```
```
After the execution of nsut steps (Lines 14-15), the agent
```
updates the weights and biases of the Target NN, ˆθ, with the
weights and biases of the Online NN, θ. Finally, the agent uses
the ultimate θ to retrieve the best actions with the highest Q-
```
values for each state (line 22).
```
We use the following structure for Online and Target
```
DNNs: four 2D convolutional layers, each followed by a batch
```
normalization layer, and a fully connected layer at the output.
The layers are using the Tanh activation function.
V. EXPERIMENTAL RESULTS
For the numerical simulations, we used MATLAB version
24.1 along with 5G Toolbox and Communications Toolbox.
Additionally, we employed the Wireless Network Simulation
Library1. Currently, MATLAB lacks support for 5G NR layers
above the RLC, i.e., RRC, PDCP, and SDAP. Therefore, func-
tionalities such as handover are not available. The simulations
were carried out on a high-performance computing cluster,
utilizing 4 nodes with 64 tasks per node and a memory
allocation of 8 GB per CPU.
We first set up a 5G scenario with one gNB and 3 slices
```
(eMBB, URLLC, and MTC). This configuration covers the
```
specification of the target scenarios in this study, where mul-
tiple services co-exist with varying requirements. We evaluate
the performance of the DRL agent with new, unseen traffic and
compare the results to those of traditional schedulers, focusing
on SLA fulfillment.
A. Experimental setup
A highly detailed system-level simulation was employed
to evaluate the solution. Table IV shows the most relevant
experimental parameters used in the numerical simulations.
```
We consider a bandwidth of 50 MHz and numerology 1 (μ=1,
```
```
subcarrier spacing of 15 kHz, 50 PRBs). Transmit power is
```
uniform across PRBs. These PRBs are shared between three
```
slices: eMBB, URLLC, and MTC. The TTI is 1 ms (1 slot),
```
i.e., there are 10 slots per 5G frame. The TDD periodicity is
5 ms, and the pattern is described in Table IV. Note that the
xApp decision is per 5G frame resolution, but the data can
be captured at a higher sampling frequency. In terms of the
application layer, the slices serve a total of 20 UEs: 5 UEs
for eMBB, 5 UEs for URLLC, and 10 UEs for MTC. The
operator configures which user belongs to which slice.
Table V shows five experimental scenarios tested with
RSLAQ, where each scenario challenges the robustness of
RSLAQ as follows:
```
1) Low traffic: In this scenario, the downlink traffic for
```
eMBB users is low, while the minimum thr per slice
configured for this slice is 10 Mbps. RSLAQ must guar-
antee the assignment of resources to comply with this
1https://uk.mathworks.com/matlabcentral/fileexchange/
119923-communications-toolbox-wireless-network-simulation-library
TABLE IV: Parameters used in the numerical simulations
Component Parameter Value Unit
gNB
```
Carrier Frequency 2.59e9 (n38) Hz
```
Duplex Mode TDD -
Channel bandwidth 10e6 Hz
Subcarrier spacing 15e3 Hz
NumRBs 50 RBs
TDD Frame 10 msDLUL periodicity 5 ms
DLUL pattern D|D|8D|4GB|4U|U|U Slot
Scheduler
CellNumber 1 -
RBAllocationLimit 50 RBs/TTI/UE
Scheduling type Slot-based -
Scheduler Slice-aware -
App
layer
Slices eMBB, URLLC,
MTC
-
```
Number of UEs 20 (5 for eMBB, 5
```
for URLLC, and 10
```
for MTC)
```
-
Physical
Layer
Full PHY with per-
fect channel esti-
mation.
```
(SISO) antenna -
```
target KPI, but at the same time, if the eMBB resources
are not used, they must be distributed between MTC and
URLLC users.
```
2) Normal: This scenario is expected during standard oper-
```
```
ation of the network; therefore, RSLAQ must be able to
```
respond according to the SLAs provided by the operator,
i.e., maintaining the target KPIs and priorities.
```
3) Congestion: This scenario is expected to occur regularly
```
with 6G applications. In this experiment, the traffic of
eMBB and MTC users is very high. RSLAQ must be
able to prioritize and optimize the use of the resources
to fulfill the SLAs by solving conflicts between them.
```
4) Stressed: This is the most challenging scenario that
```
RSLAQ can face, where, in addition to network con-
gestion, the SLA provided by the operator has stringent
requirements. To represent this, the minimum thr for
eMBB users is elevated compared to previous experi-
ments.
```
5) Insufficient resources: The design of RSLAQ assumes
```
that the SLA must align with the resource allocation
established in a previous stage at the Non-RT RIC
```
(SMO). However, it is important to analyze the behavior
```
of RSLAQ in this scenario. This allows us to generate
safeguard mechanisms based on RSLAQ outputs, con-
sidering that there could be instances where the Non-
RT RIC can misconfigure the SLA requirements with
respect to the available resources. These mechanisms
contribute to the overall system robustness.
B. DRL training
To improve model reliability and ensure stable conver-
gence, we conducted a hyperparameter sensitivity analysis.
Six different hyperparameter configurations were explored for
scenario 1, as shown in Table VI. Each set introduces targeted
variations in learning rate, discount factor, exploration decay,
and buffer size to identify the best trade-off between stability,
convergence speed, and sampling efficiency. The learning
curves for these configurations are shown in Fig. 4, from which
This article has been accepted for publication in IEEE Transactions on Mobile Computing. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TMC.2026.3666787
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: UNIVERSIDADE FEDERAL DO PARA. Downloaded on February 26,2026 at 18:13:50 UTC from IEEE Xplore. Restrictions apply.
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 11
TABLE V: Experimental conditions: varying network conditions and SLAs.
Experiment Network condition SLAs
```
Low traffic eMBB= 50 Kbps; URLLC= 1 Mbps;
```
```
MTC= 2 Mbps;
```
```
eMBB={min thr = 10 Mbps; max thr = 15 Mbps; Priority = 2, Maximize thr}; URLLC={max
```
```
bfs = 3%; Priority = 1, Minimize bfs}; MTC ={No Policy ; Priority = 3; Maximize thr}
```
```
Normal eMBB= 70 Mbps; URLLC= 1 Mbps;
```
```
MTC= 2 Mbps;
```
```
eMBB={min thr =10 Mbps; max thr = 15 Mbps; Priority = 2, Maximize thr}; URLLC={max
```
```
bfs = 3%; Priority = 1, Minimize bfs}; MTC={No Policy; Priority = 3; Maximize thr}
```
```
Congestion eMBB= 100 Mbps; URLLC= 1 Mbps;
```
```
MTC= 100 Mbps
```
```
eMBB={min thr = 10 Mbps; max thr = 15 Mbps; Priority = 2; Maximize thr}; URLLC={max
```
```
bfs = 3%; Priority = 1; Minimize bfs}; MTC={No Policy; Priority = 3; Maximize thr}
```
```
Stressed eMBB= 100 Mbps; URLLC= 1 Mbps;
```
```
MTC= 100 Mbps
```
```
eMBB={min thr = 20 Mbps; max thr = 25 Mbps; Priority = 2, Maximize thr}; URLLC={max
```
```
bfs = 3%; Priority = 1; Minimize bfs}; MTC={No Policy; Priority = 3; Maximize thr}
```
Insufficient
resources
```
eMBB= 100 Mbps; URLLC= 2 Mbps;
```
```
MTC= 100 Mbps
```
```
eMBB={min thr = 20 Mbps; max thr = 25 Mbps; Priority = 2; Maximize thr}; URLLC={max
```
```
bfs = 3%; Priority = 1; Minimize bfs}; MTC={No Policy; Priority = 3; Maximize thr}
```
00.5
1
00.5
1
00.5
1
00.5
1
00.5
1
0 500 1000 1500 2000 2500 3000 3500
00.5
1
Fig. 4: Hyperparameter rewards. Hyp-set3 is selected.
Hyp-set3 was selected, as it demonstrates correct convergence
and maintains stability after reaching convergence.
TABLE VI: Hyperparameter variations across six configura-
tions. Fixed parameters: pstat = 0.5, nsut = 200, ntsr = 100,
```
λϵ = 0.9, ϵmin = 0.05, L = 500 (except where noted),
```
```
btsz = 350 (except where noted). LR: learning rate of DNNs.
```
Parameters Description
Hyp-set 1: LR = 0.001, γ = 0.80 Moderate learning speed,
short-term reward focus.
```
Hyp-set 2: LR = 0.001, γ = 0.90 Higher γ; better long-
```
term rewards.
Hyp-set 3: LR = 0.001, γ = 0.80, λϵ
= 0.998
```
Faster decay; quicker ex-
```
ploitation.
```
Hyp-set 4: LR = 0.0005, γ = 0.90 Lower LR + Higher γ;
```
more stable.
```
Hyp-set 5: LR = 0.002, γ = 0.85 Higher LR + Higher γ;
```
Faster convergence.
Hyp-set 6: LR = 0.001, γ = 0.85,
```
btsz = 400, L = 600, λϵ = 0.9995
```
Larger Buffer + Slower
```
Decay; Better sampling.
```
```
Fig. 5 shows the average reward in the learning steps (1 step
```
```
= 10 ms frame) for all scenarios using Hyp-set3. In general,
```
in scenarios 1-4, after an initial exploration phase, the system
successfully converges to a solution that minimizes outages
and soft KPI violations. Scenarios 1-4 are incrementally com-
plex to solve in terms of network conditions and SLAs, and
RSLAQ shows robustness to learn in these different scenarios.
0 500 1000 1500 2000 2500 3000 3500
-0.5
0
0.5
1
1.5
2
Fig. 5: Reward function of RSLAQ for different SLAs and
network conditions.
```
In scenario 4 (stressed), which combines network congestion
```
and stringent SLA requirements, RSLAQ finds it difficult to
enforce the SLA policy provided by the operator. Nevertheless,
it achieves optimal conditions after a number of steps.
The reward function is designed to produce negative values
if RSLAQ cannot fulfill outage SLAs. Therefore, if the radio
resources are insufficient to cover SLA specifications, RSLAQ
```
reward values are negative (see Insufficient resources in Fig.
```
```
5). In particular, we observed that in our experiments for
```
scenario 5, the traffic for URLLC is very high and the available
resources cannot meet the minimum bfs per URLLC UE,
```
even when RSLAQ allocates all available resources (50 PRBs)
```
exclusively to this slice. If this event is detected, RSLAQ sends
an alarm to the Non-RT RIC, which should promptly correct
the misconfiguration. In the meantime, RSLAQ can revert to a
previous configuration or any safety condition established by
the operator to ensure the QoS for users is maintained.
C. DRL testing
We compare the performance of RSLAQ against five base-
lines. Three are standard schedulers: RR, PF, and BCQI. The
fourth baseline is an optimization-based mechanism that allo-
```
cates resources per slice (Opt), as presented in [22], reflecting
```
prior work focused on maximizing resource efficiency. Finally,
This article has been accepted for publication in IEEE Transactions on Mobile Computing. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TMC.2026.3666787
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: UNIVERSIDADE FEDERAL DO PARA. Downloaded on February 26,2026 at 18:13:50 UTC from IEEE Xplore. Restrictions apply.
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 12
we include the SAC approach proposed in [27], which shares
RSLAQ’s philosophy of optimizing resource allocation while
ensuring SLA compliance.
Some considerations were made when implementing
SAC [27] to enable a direct comparison with RSLAQ. First,
the state definition was aligned with that used for RSLAQ.
```
Notably, SAC [27] does not optimize intra-slice schedulers;
```
instead, the proportional fairness scheduler is fixed across
all scenarios. The action space in SAC [27] is continuous,
```
representing resource distribution among slices (pj ). To ensure
```
validity without discretization, the constraint Pj pj = 1 is
enforced by using a softmax function. These outputs are
interpreted as probabilities by the slice-aware scheduler and
applied to each slice. The same slice-aware scheduler proposed
for RSLAQ was used, ensuring that resources are not allocated
to slices without traffic demand but re-distributed to active
slices, maintaining fairness in comparison.
Regarding the reward function, SAC [27] uses a similar
formulation to RSLAQ, with key differences: outage condi-
tions and No-Policy slices are not considered. To incorporate
the non-policy MTC slice into the SAC approach, the reward
```
was adapted by maintaining its maximum value at zero (as
```
```
originally defined) and adding the normalized and weighted
```
MTC throughput to the formula.
Finally, multiple simulations of SAC [27] were conducted
using different hyperparameter configurations to identify the
best-performing setup. Specifically, variations were explored
in SAC hyperparameters: learning rate, entropy coefficient,
target smoothing factor, batch size, and replay buffer size. The
best-performing configuration for SAC [27] is as follows. The
SAC agent uses an actor with a convolutional backbone and
```
two heads (logits and log-std) for a continuous 3-D action
```
```
space (pj ) normalized by softmax to satisfy Pj pj = 1,
```
and a critic combining state and action streams via fully
connected layers. Training employed Adam optimizer with
learning rate 10−3, discount factor for future rewards 0.99,
entropy coefficient 0.1, target smoothing factor 0.005, batch
size 256, and replay buffer size 10,000. As described, As
described, we acknowledge that our implementation and test-
ing of SAC were conducted under hyperparameters, SLAs,
simulation parameters, and network scenarios that are not
identical to those in [27]. Our goal was to validate the core
concept of SAC [27] under stressed conditions that can occur
in practical 5G/6G deployments, where robustness is required.
Fig. 6 compares the performance of the five baselines vs.
```
RSLAQ. In the first scenario (Fig. 6 (a)), RSLAQ allocates the
```
resources following SLA specifications for eMBB and URLLC
and achieves maximum thr for MTC. Other approaches cannot
comply with target KPIs, particularly with eMBB. Note that
RSLAQ allocates resources for eMBB, but since the traffic
```
for eMBB is low (see Table V, first row) these resources
```
are allocated to URLLC and MTC users. In this scenario, the
double control loop in our framework helps allocate unused
resources. SAC maintains SLA compliance for URLLC, but
overallocates resources to eMBB.
```
In the second scenario, in Fig. 6 (b), the users of all
```
```
slices behave as expected under normal conditions (high
```
throughput for eMBB, a higher number of devices with low
```
throughput for MTC and low throughput for URLLC). Unlike
```
other mechanisms, RSLAQ is able to optimally distribute the
resources to fulfil SLAs. SAC also fulfills SLAs for URLLC,
with over-allocation of resources for eMBB. If SLAs are
not considered, while some KPIs may show improvement
with certain schedulers, they often come at the cost of other
important KPIs. For example, in the case of the optimized
```
approach shown in Fig. 6 (b), the thr for MTC is significantly
```
better than that of RSLAQ. However, under these conditions,
the outage KPI for URLLC is violated.
```
The third scenario (congestion) in Fig. 6 (c) is also frequent
```
in 5G/6G. In this scenario, the downlink traffic for MTC and
eMBB users is very high. Prioritization comes into play in this
scenario, where MTC has the lowest priority, and URLLC the
highest priority. RSLAQ properly allocates the resources and
selects the schedulers to fulfill the SLAs, as seen in Fig. 6
```
(c). Similarly, in this scenario, SAC significantly outperforms
```
other baselines, although it tends to over-allocate resources to
eMBB.
```
In the stressed scenario (see Fig. 6 (d)), the robustness limits
```
of RSLAQ are tested, as in addition to network congestion,
the SLAs are very strict. In particular, eMBB target KPIs are
increased. Under these conditions, there are instances where
outages cannot be avoided, as the resources cannot cover all
SLAs at the same time. RSLAQ gives priorities to which
outages should be minimized, and in our experiments, it is the
URLLC. The weights in ωj provide flexibility in controlling
```
the behavior of RSLAQ in this scenario. As seen in Fig. 6 (d),
```
the outage SLAs for eMBB and URLLC are minimized and the
resources allocated to MTC are very low. SAC ensures better
SLA compliance for URLLC, but this comes at the expense
of eMBB performance.
Finally, we also present the results for the scenario in which
the resources are insufficient to fulfill outage SLAs in Fig.
```
6 (e). In this scenario, the target KPI for URLLC cannot
```
be met with the resource available for dynamic allocation
in the system. RSLAQ cannot guarantee an optimal resource
distribution in this scenario, as the allocation of resources to
```
the system (50 PRBs) cannot be modified at the Near-RT RIC
```
level. RSLAQ logs this event and notifies the Non-RT RIC
which should re-configure the system in terms of the resource
allocation and SLAs. In this last scenario, SAC achieves SLA
compliance for the slice with the highest priority. Nevertheless,
this scenario represents an outage condition for the system.
```
Overall, as shown in Figs. 6 (a)-(d), the optimized approach
```
```
(Opt) outperforms RR, PF, and BCQI by simultaneously
```
maximizing thr for eMBB and MTC while minimizing bfs for
```
URLLC (considering priorities ωj ). However, this approach
```
still fails to meet all SLAs. SAC outperforms Opt as it incor-
porates SLA considerations into the optimization. However, it
tends to over-allocate resources to eMBB due to the absence
of soft-KPI constraints. RSLAQ not only ensures that the
distribution aligns with SLAs but also guarantees optimized
resource allocation across different traffic conditions. Our so-
```
lution demonstrates that connecting the application plane (user
```
```
policy) and the data plane (channel) through the intelligent
```
QoS xApp in the O-RAN RIC significantly improves resource
utilization, benefiting both the users and the network operator.
This article has been accepted for publication in IEEE Transactions on Mobile Computing. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TMC.2026.3666787
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: UNIVERSIDADE FEDERAL DO PARA. Downloaded on February 26,2026 at 18:13:50 UTC from IEEE Xplore. Restrictions apply.
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 13
0 5 10 15 20 25 30 350
0.2
0.4
0.6
0.8
1
RRPF
BCQIOpt [22]
SAC [27]RSLAQ
target KPISLA align
Outage
0 5 10 15 200
0.2
0.4
0.6
0.8
1
Better
RRPF
BCQIOpt [22]
SAC [27]RSLAQ
0 20 40 60 80 1000
0.2
0.4
0.6
0.8
1
RRPF
BCQIOpt [22]
SAC [27]RSLAQ
target KPISLA align
Outage
```
(a) Low traffic
```
0 5 10 15 20 25 30 350
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1
RRPF
BCQIOpt [22]
SAC [27]RSLAQ
target KPISLA align
Outage
0 5 10 15 200
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1
Better
RRPF
BCQIOpt [22]
SAC [27]RSLAQ
0 20 40 60 80 1000
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1
RRPF
BCQIOpt [22]
SAC [27]RSLAQ
target KPISLA align
Outage
```
(b) Normal traffic
```
0 5 10 15 20 25 30 350
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1
RRPF
BCQIOpt [22]
SAC [27]RSLAQ
target KPISLA align
Outage
0 5 10 15 200
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1
Better
RRPF
BCQIOpt [22]
SAC [27]RSLAQ
0 20 40 60 80 1000
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1
RRPF
BCQIOpt [22]
SAC [27]RSLAQ
target KPISLA align
Outage
```
(c) Congestion
```
0 5 10 15 20 25 30 350
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1
RRPF
BCQIOpt [22]
SAC [27]RSLAQ
target KPISLA align
Outage
0 5 10 15 200
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1
Better
RRPF
BCQIOpt [22]
SAC [27]RSLAQ
0 20 40 60 80 1000
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1
RRPF
BCQIOpt [22]
SAC [27]RSLAQ
target KPISLA align
Outage
```
(d) Stressed
```
0 5 10 15 20 25 30 350
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1
RRPF
BCQIOpt [22]
SAC [27]RSLAQ
target KPISLA align
Outage
0 5 10 15 200
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1
Better
RRPF
BCQIOpt [22]
SAC [27]RSLAQ
0 20 40 60 80 1000
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1
RRPF
BCQIOpt [22]
SAC [27]RSLAQ
target KPISLA align
Outage
```
(e) Insufficient resources
```
```
Fig. 6: RSLAQ performance vs. other mechanisms. RSLAQ enforces target KPIs (SLAs) in different traffic and SLA conditions
```
```
considered in the design (sufficient resources). When the system resources are insufficient, URLLC cannot be fulfilled.
```
This article has been accepted for publication in IEEE Transactions on Mobile Computing. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TMC.2026.3666787
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: UNIVERSIDADE FEDERAL DO PARA. Downloaded on February 26,2026 at 18:13:50 UTC from IEEE Xplore. Restrictions apply.
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 14
Opt [22] SAC [27] RSLAQ0
0.2
0.4
0.6
0.8
1
Reliability for eMBBReliability for URLLC
Opt [22] SAC [27] RSLAQ0
0.2
0.4
0.6
0.8
1
Reliability for eMBBReliability for URLLC
Opt [22] SAC [27] RSLAQ0
0.2
0.4
0.6
0.8
1
Reliability for eMBBReliability for URLLC
Opt [22] SAC [27] RSLAQ0
0.2
0.4
0.6
0.8
1
Reliability for eMBBReliability for URLLC
Fig. 7: Outage probability of RSLAQ vs. the optimized approach.
VI. DISCUSSION, LIMITATIONS, AND FUTURE WORK
This section highlights the flexibility, robustness, and relia-
bility benefits of RSLAQ, explores potential improvements,
addresses its limitations, and suggests directions for future
work.
A. Robustness and reliability
In terms of robustness, we have tested four different scenar-
```
ios (varying network conditions and SLAs) for which RSLAQ
```
demonstrated consistent performance in SLA fulfillment. Note
that the design of the reward function is self-explanatory and
enables rapid assessment of whether the system has reached
a feasible solution or is unable to find one. If the long-
term reward remains below zero at the end of the training,
no feasible solution was identified. Note that this scenario
only occurs if the available resources and SLAs are not
properly validated at the Non-RT RIC. RSLAQ incorporates
fail-safe mechanisms that include the activation of default
3GPP schedulers, reverting the system to a previous condition,
```
or using default slicing strategies (e.g., reserved slices) to
```
maintain system performance until RSLAQ is re-trained with
different configurations of resources and SLAs.
A key requirement for 6G communications is high reliabil-
ity, especially critical for URLLC. Fig. 7 compares the average
```
reliability of RSLAQ, calculated as (1 − P (kout)), with the
```
```
optimized approach (Opt). The standard schedulers PF, RR,
```
and BCQI have been excluded from this analysis, as they
are QoS-unaware. Furthermore, the scenario of insufficient
resources is not considered, as RSLAQ operates under the
assumption that the Non-RT RIC guarantees that the existing
resources cover the SLA requirements.
As shown in Fig. 7, RSLAQ achieves the highest simulta-
neous reliability for eMBB and URLLC. For the first three
scenarios, which are expected to cover most of the network
```
conditions, RSLAQ achieves a high reliability rate (> 95%).
```
This value is reduced to approximately 80% when the network
is congested and the operator has allocated resources and SLAs
at their maximum allowed limits. SAC [27] also achieves
high reliability, outperforming the Opt approach under all
conditions. However, since robustness is not central to its
design, it cannot match the reliability delivered by RSLAQ.
We also present a sensitivity analysis for RSLAQ by varying
the proportion of static allocation of PRB resources, psta,
Low traffic Normal traffic Congestion Stressed0
0.2
0.4
0.6
0.8
1
```
Fig. 8: Outage probability with inverted priorities (ωj =
```
```
[0.40, 0.33, 0.27]).
```
to assess its effect on system reliability. Table VII presents
reliability values for eMBB and URLLC under six configu-
rations of psta. Highlighted pairs indicate the best option per
scenario, chosen where URLLC is higher than eMBB and the
```
combined average is highest. While higher psta (0.5 − 0.7)
```
generally yield better averages, some configurations such as
```
psta = 0.7, though strong for most cases, produce eMBB
```
reliability values far from the expected ideal levels, especially
for the critical stressed scenario. This sensitivity analysis
shows psta = 0.5 and psta = 0.6 offer a more balanced trade-
off, maintaining high URLLC reliability while keeping eMBB
closer to desirable values. Conversely, very low static resource
```
allocation (e.g., psta = 0.1) perform poorly overall, particularly
```
for URLLC in challenging conditions, reinforcing that mid-
range psta are the most robust choices. In general, low values
of psta are not recommended because they fail to achieve the
high reliability targets expected for both services, making them
unsuitable for stringent SLA requirements.
Additionally, Fig. 8 shows the effect of changing slice
priorities, assigning higher priority to eMBB over URLLC,
```
with weights [0.40, 0.33, 0.27] (priorities [1, 2, 3]). RSLAQ
```
maintains high reliability for both slices while preserving
higher priority levels for eMBB in critical scenarios, such as
congestion and stressed conditions.
B. Complexity and scalability
The complexity of RSLAQ is as follows. At each training
```
step, RSLAQ performs: (i) state construction; (ii) reward
```
This article has been accepted for publication in IEEE Transactions on Mobile Computing. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TMC.2026.3666787
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: UNIVERSIDADE FEDERAL DO PARA. Downloaded on February 26,2026 at 18:13:50 UTC from IEEE Xplore. Restrictions apply.
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 15
```
TABLE VII: Reliability for different scenarios and weights (best pair highlighted).
```
Weight Low traffic Normal traffic Congestion Stressed
eMBB URLLC eMBB URLLC eMBB URLLC eMBB URLLC
```
psta = 0.1 0.908 0.954 0.927 0.819 0.949 0.996 0.854 0.001
```
```
psta = 0.2 0.920 0.964 0.941 0.939 0.938 0.968 0.901 0.006
```
```
psta = 0.4 0.947 0.995 0.953 0.940 0.953 0.990 0.764 0.777
```
```
psta = 0.5 0.957 0.993 0.952 0.960 0.972 0.976 0.832 0.783
```
```
psta = 0.6 0.948 0.997 0.950 0.968 0.950 0.962 0.842 0.853
```
```
psta = 0.7 0.954 0.996 0.961 0.987 0.952 0.983 0.733 0.951
```
```
computation; and (iii) DDQL training. (i) and (ii) are linear
```
```
in the number of slices J and UEs N , i.e., O(JN ), since
```
each slice aggregates per-UE KPIs. The DDQL update is
dominated by the neural network forward/backward passes.
```
Let the network have σ layers with sizes {n0, . . . , nσ }, where
```
the final layer equals the action space, nσ = |A|. The per-
step forward cost is F = O
 Pσ
```
l=1 nl−1nl
```

, and the
```
backward/update cost is of the same order, U = Θ(F ).
```
Because nσ = |A|, both F and U grow linearly with
the action space through the last layer. Over E episodes
and T steps per episode, the total training complexity is
O
 
```
E T (JN + F + U )
```

= O

E T

JN + Pσl=1 nl−1nl

,
where the Pl nl−1nl term is typically dominated by the output
layer contribution nσ−1|A|, making the dependence on |A| the
primary driver of training time. Inference per decision step
is a single forward pass, O
  Pσ
```
l=1 nl−1nl
```

, and thus remains
lightweight, scaling linearly with |A| via the final layer.
According to this analysis, RSLAQ scales to any number
of UEs. In addition, when the number of slices increases
```
(i.e., > 3), the design of RSLAQ can be flexibly adapted as
```
explained next. The number of actions increases proportionally
to the number of slices. Since the action space defines the
number of neurons in the output layer of the DNN in the
DRL architecture, a change in the number of slices changes
the DNN design. In terms of the reward function, as new slices
are defined, new SLAs appear. The calculation of the reward
is based on the SLAs and thus the reward function should
include them. With respect to performance, the demands on
memory, processing resources, and learning time increase as
more scenarios need to be evaluated. Nevertheless, the runtime
performance of the DNN is expected to be maintained once the
model has been trained. With these modifications, offline re-
training is required. Offline refers to isolating RL updates from
```
the live system during re-training; once training is complete,
```
the RL agent can be re-integrated into production. Note that
this can be automated for deployment. A very high number of
slices may hinder DRL learning. Coarse-grained discretization
of pj can help reduce the complexity of the learning process.
Finally, the proposed approach can scale to multiple cells
```
(RUs/DUs/CUs) within the same RAN by either deploying
```
an RL agent per cell or aggregating states for a centralized
agent. The choice between distributed and centralized control
depends on deployment requirements and is supported by the
O-RAN architecture.
C. Limitations and future work
```
1) Enhancements to RSLAQ: To achieve a more robust and
```
faster learning convergence of the DRL, the design necessarily
limited the action space, particularly the resolution of the
proportion of PRBs/slice. As shown in Fig. 7, the DRL
successfully met the performance objectives despite this low
resolution. Lower resolution reduces resource needs for model
training and deployment, resulting in lower energy consump-
tion. Nevertheless, depending on the computational capability
available, this resolution can be increased to enhance perfor-
mance. However, note that increasing the resolution signif-
icantly increases the complexity of the neural network. For
example, using the 0.01 resolution results in A ∈ R15453X4,
making the size of A 78 times larger than with a 0.1 resolution.
This design could result in longer learning times and possibly
limited exploration of certain action-state spaces due to the
infinite number of possible states. In the RSLAQ design, we
prioritized fast learning and high reliability.
Actor-critic agents with continuous outputs can also be
applied to our model, provided that the selected actions are
constrained to valid resource allocations. One example of
how this can be achieved is presented in [27] where the
continuous output of the actor network is mapped to the closest
```
valid allocation from a pre-defined set (finite) of feasible
```
combinations. Note that although the RL agent can output
highly granular or continuous values for PRB distribution
across slices, the scheduler enforces discrete allocation. Each
PRB represents the smallest indivisible unit, so resources are
assigned in whole numbers, not fractional values. A detailed
experimentation and validation of these methods is left for
future work.
Another important future research direction is the integra-
tion of dApps for real-time inference and control in O-RAN,
as proposed in [37]. dApps can enable distributed, low-latency
decision-making for functions such as per-slice scheduling,
complementing the xApp approach considered in this work.
```
2) Additional SLAs: Integrating an admission control mech-
```
```
anism (e.g., handovers) could enhance the ability of the system
```
to manage SLAs and accommodate additional SLA options,
such as the maximum number of UEs per slice. Furthermore,
although we do not explicitly include an energy model in the
system, the efficient distribution of PRBs contributes to better
energy utilization. Currently, RSLAQ cannot control trans-
mission power. The implementation of these features requires
protocols at the RRC layer and functionalities within the Non-
RT RIC, which are beyond the scope of this study. The model
can also be extended to support percentile guarantees, penalty
This article has been accepted for publication in IEEE Transactions on Mobile Computing. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TMC.2026.3666787
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: UNIVERSIDADE FEDERAL DO PARA. Downloaded on February 26,2026 at 18:13:50 UTC from IEEE Xplore. Restrictions apply.
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 16
curves, and multi-tenant bargaining, in order to better align
with real-world SLA contracts. These aspects can be explored
in future research.
```
3) Testbed validation and multi-scenario analysis: Future
```
work will focus on deploying RSLAQ in a real testbed to
measure inference latency, CPU utilization, and telemetry
overhead. Future research will also examine variations of com-
plex scenarios, such as uplink-heavy or mmWave deployments,
multi-cell interference, and mobility-induced handovers, where
stringent SLAs and additional constraints are imposed.
```
4) Security of ML in O-RAN: The use of ML in O-
```
RAN introduces new vectors for adversarial attacks [38].
The softwarization and openness of O-RAN facilitate the
deployment of rogue elements. When deploying RSLAQ in
real-world scenarios, robust authorization and authentication
mechanisms for UEs, E2 nodes, and third-party xApps must be
employed to prevent intrusions that could severely compromise
the performance of the system. For instance, rogue UEs
```
could introduce misleading channel state information (CSI),
```
compromising the normal operation of the schedulers and
negatively affecting the learning and performance of the DRL
agent. Additionally, malicious xApps might corrupt KPMs
```
stored in the RIC database (such as thr, bfs, and UEs/slice),
```
undermining the inputs of the DRL method and thus com-
promising SLA compliance. Future research will focus on
```
developing reactive (e.g., anomaly detection) and proactive
```
```
(e.g., adversarial learning) defense mechanisms to ensure the
```
robustness of DRL systems against such adversarial attacks.
VII. CONCLUSION
A reliable DRL-based QoS xApp, RSLAQ, was designed
and tested using detailed simulations. Our design using 3GPP
standard mechanisms, such as using RR, P F , and BCQI
schedulers, allows seamless integration of RSLAQ into exist-
ing 5G/6G systems without significant modifications. Experi-
mental results show that incorporating QoS awareness into the
RAN system not only maximizes KPIs for each type of service
but also ensures compliance with specific SLAs. RSLAQ
presented consistent performance across a diverse range of
scenarios, demonstrating its robustness and adaptability in
varying network conditions. This approach benefits both the
users by maintaining their QoS and the network operator,
as it enables the deployment of additional services without
compromising the agreed quality of existing ones.
ACKNOWLEDGMENTS
This work is supported through the NICYBER2025 pro-
gramme funded by Innovate UK. The ORANSecAI project is
a collaboration with Ampliphae. The views expressed are those
of the authors and do not necessarily represent the project or
the funding agency.
REFERENCES
[1] K. Park, S. Sung, H. Kim, and J. il Jung, “Technology trends and chal-
lenges in SDN and service assurance for end-to-end network slicing,”
Computer Networks, vol. 234, p. 109908, 2023.
```
[2] O-RAN, “O-RAN Work Group 1 (Use Cases and Overall Architecture)
```
```
Use Cases Analysis Report (O-RAN.WG1.Use-Cases-Analysis-Report-
```
```
R003-v12.00),” Available at https://www.o-ran.org/ (2024/05/23), O-
```
RAN Alliance, Tech. Rep., 2023.
[3] R. Sghaier, C. El Hog, R. Ben Djemaa, and L. Sliman, “A Review on
SLA Monitoring Based on Blockchain,” in Intelligent Systems Design
and Applications. Springer, 2024, pp. 458–467.
[4] N. Yungaicela-Naula, V. Sharma, and S. Scott-Hayward, “SLAQ: An
SLA-Driven 6G O-RAN QoS Framework Using Deep Reinforcement
Learning,” in 2025 International Symposium on Networks, Computers
```
and Communications (ISNCC), 2025, pp. 1–6.
```
```
[5] 3GPP, “3GPP TS 23.501; Technical Specification Group Services and
```
```
System Aspects; System architecture for the 5G System (5GS); Stage
```
```
2 (release 18),” Available at https://www.3gpp.org/ (2024/07/16), 3GPP,
```
Tech. Rep., 2024.
[6] C. Cox, An introduction to 5G: the new radio, 5G network and beyond.
John Wiley & Sons, 2020.
[7] M. Irazabal and N. Nikaein, “TC-RAN: A Programmable Traffic Control
Service Model for 5G/6G SD-RAN,” IEEE Journal on Selected Areas
in Communications, vol. 42, no. 2, pp. 406–419, 2024.
```
[8] O-RAN, “O-RAN Working Group 1 (Use Cases and Overall Archi-
```
```
tecture) O-RAN Architecture Description (O-RAN.WG1.OAD-R003-
```
```
v09.00),” Available at https://www.o-ran.org/ (2024/05/23), O-RAN
```
Alliance, Tech. Rep., 2023.
[9] S. Sirotkin, 5G Radio Access Network Architecture: The Dark Side of
5G. John Wiley & Sons, 2020.
```
[10] O-RAN, “O-RAN Working Group 3 (Near-Real-time RAN Intelligent
```
```
Controller and E2 Interface Workgroup) (O-RAN.WG3.RICARCH-
```
```
R003-v04.00),” Available at https://www.o-ran.org/ (2024/05/23), O-
```
RAN Alliance, Tech. Rep., 2023.
```
[11] ——, “O-RAN Work Group 1 (Use Cases and Overall Archi-
```
```
tecture). Use Cases Detailed Specification (O-RAN.WG1.Use-Cases-
```
```
Analysis-Report-R003-v12.00),” Available at https://www.o-ran.org/
```
```
(2024/05/23), O-RAN Alliance, Tech. Rep., 2023.
```
```
[12] O-RAN, “O-RAN Work Group 2 (Non-RT RIC & A1/R1 interface):
```
```
Use Cases and Requirements (O-RAN.WG2.Use-Case-Requirements-
```
```
R003-v10.00),” Available at https://www.o-ran.org/ (2024/07/19), O-
```
RAN Alliance, Tech. Rep., 2024.
[13] L. Bonati, S. D’Oro, M. Polese, S. Basagni, and T. Melodia, “In-
telligence and Learning in O-RAN for Data-Driven NextG Cellular
Networks,” IEEE Communications Magazine, vol. 59, no. 10, pp. 21–27,
2021.
[14] M. Zambianco and G. Verticale, “A reinforcement learning agent for
mixed-numerology interference-aware slice spectrum allocation with
non-deterministic and deterministic traffic,” Computer Communications,
vol. 189, pp. 100–109, 2022.
[15] A. Papa, P. Kutsevol, F. Mehmeti, and W. Kellerer, “Effects of SD-RAN
Control Plane Design on User Quality of Service,” in 2022 IEEE 8th
```
International Conference on Network Softwarization (NetSoft), 2022, pp.
```
312–320.
[16] Y.-C. Jian, M.-S. Chung, H. Susanto, and F.-Y. Leu, “5G Base Station
Scheduling,” in Innovative Mobile and Internet Services in Ubiquitous
Computing, L. Barolli, Ed. Cham: Springer International Publishing,
2022, pp. 315–324.
[17] F. Lotfi, O. Semiari, and F. Afghah, “Evolutionary deep reinforcement
learning for dynamic slice management in o-ran,” in 2022 IEEE Globe-
```
com Workshops (GC Wkshps), 2022, pp. 227–232.
```
[18] Y. Yang and X. Sindhu, “Dynamic RAN Slice Resource Allocation
for SLA Assurance - O-RAN Plugfest Spring 2022,” 2022. [Online].
```
Available: https://plugfestvirtualshowcase.o-ran.org/2022/SPRING
```
[19] KDDI and Samsung, “Field Trials of latency assurance with Near-RT
RIC for E2E NW Slicing - O-RAN Plugfest Fall 2022,” 2022. [Online].
```
Available: https://plugfestvirtualshowcase.o-ran.org/2022/FALL
```
[20] C. C. Zhang, K. K. Nguyen, and M. Cheriet, “Joint routing and packet
scheduling for urllc and embb traffic in 5g o-ran,” in ICC 2022 - IEEE
International Conference on Communications, 2022, pp. 1900–1905.
[21] ONF and R. Labs, “Policy-controlled QoS-based Resource
```
Allocation xApp (QRA-xApp) from Rimedo Labs with
```
ONF’s SD-RAN RIC,” 2023. [Online]. Available: https:
//www.virtualexhibition.o-ran.org/classic/generation/2024/category/
intelligent-ran-control-demonstrations/sub/intelligent-control/238
[22] Tsampazi, Maria and D’Oro, Salvatore and Polese, Michele and Bon-
ati, Leonardo and Poitau, Gwenael and Healy, Michael and Melodia,
Tommaso, “A Comparative Analysis of Deep Reinforcement Learning-
based xApps in O-RAN,” in Proceedings of IEEE GLOBECOM, Kuala
Lumpur, Malaysia, December 2023.
This article has been accepted for publication in IEEE Transactions on Mobile Computing. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TMC.2026.3666787
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: UNIVERSIDADE FEDERAL DO PARA. Downloaded on February 26,2026 at 18:13:50 UTC from IEEE Xplore. Restrictions apply.
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 17
[23] T. Vladic and M. Jones, “RAN Control of UE-specific DRB-level QoS
attributes - O-RAN Plugfest Fall 2023,” 2023. [Online]. Available:
```
https://plugfestvirtualshowcase.o-ran.org/2023/FALL
```
[24] F. Rezazadeh, L. Zanzi, F. Devoti, H. Chergui, X. Costa-P´erez, and
C. Verikoukis, “On the Specialization of FDRL Agents for Scalable
and Distributed 6G RAN Slicing Orchestration,” IEEE Transactions on
Vehicular Technology, vol. 72, no. 3, pp. 3473–3487, 2023.
[25] M. Sulaiman, M. Ahmadi, M. A. Salahuddin, R. Boutaba, and A. Saleh,
“Generalizable Resource Scaling of 5G Slices using Constrained Rein-
forcement Learning,” in NOMS 2023-2023 IEEE/IFIP Network Opera-
tions and Management Symposium, 2023, pp. 1–9.
[26] M. Zangooei, M. Golkarifard, M. Rouili, N. Saha, and R. Boutaba,
“Flexible RAN Slicing in Open RAN With Constrained Multi-Agent
Reinforcement Learning,” IEEE Journal on Selected Areas in Commu-
nications, vol. 42, no. 2, pp. 280–294, 2024.
[27] C. V. Nahum, V. H. L. Lopes, R. M. Dreifuerst, P. Batista, I. Correa,
K. V. Cardoso, A. Klautau, and R. W. Heath, “Intent-Aware Radio
Resource Scheduling in a RAN Slicing Scenario Using Reinforcement
Learning,” IEEE Transactions on Wireless Communications, vol. 23,
no. 3, pp. 2253–2267, 2024.
[28] R. Raftopoulos, S. D’Oro, T. Melodia, and G. Schembra, “DRL-
Based Latency-Aware Network Slicing in O-RAN with Time-Varying
SLAs,” in 2024 International Conference on Computing, Networking
```
and Communications (ICNC), 2024, pp. 737–743.
```
[29] Y. Huang, Q. Sun, N. Li, Z. Chen, J. Huang, H. Ding, and C.-L. I,
“Validation of Current O-RAN Technologies and Insights on the Future
Evolution,” IEEE Journal on Selected Areas in Communications, vol. 42,
no. 2, pp. 487–505, 2024.
[30] J. Dai, L. Li, R. Safavinejad, S. Mahboob, H. Chen, V. V. Ratnam,
H. Wang, J. Zhang, and L. Liu, “O-ran-enabled intelligent network
```
slicing to meet service-level agreement (sla),” IEEE Transactions on
```
Mobile Computing, vol. 24, no. 2, pp. 890–906, 2025.
[31] A. M. Nagib, H. Abou-Zeid, and H. S. Hassanein, “SafeSlice: Enabling
SLA-Compliant O-RAN Slicing via Safe Deep Reinforcement Learn-
ing,” in 2025 IEEE International Conference on Machine Learning for
```
Communication and Networking (ICMLCN), 2025, pp. 1–7.
```
[32] R. Barker, A. E. Dorcheh, T. Seyfi, and F. Afghah, “REAL: Re-
inforcement Learning-Enabled xApps for Experimental Closed-Loop
Optimization in O-RAN with OSC RIC and srsRAN,” arXiv preprint
```
arXiv:2502.00715, 2025.
```
[33] G. Kougioumtzidis, A. Vlahov, V. K. Poulkov, P. I. Lazaridis, and
Z. D. Zaharis, “QoE Prediction for Gaming Video Streaming in O-
RAN Using Convolutional Neural Networks,” IEEE Open Journal of
the Communications Society, vol. 5, pp. 1167–1181, 2024.
[34] M. Yassin, W. Diego, and S. Imadali, “Demo Abstract: 5G End-to-End
Open Source Network with Traffic Prioritization Mechanism,” in IEEE
INFOCOM 2019 - IEEE Conference on Computer Communications
```
Workshops (INFOCOM WKSHPS), 2019, pp. 973–974.
```
[35] M. Polese, L. Bonati, S. D’Oro, S. Basagni, and T. Melodia, “ColO-
```
RAN: Developing Machine Learning-Based xApps for Open RAN
```
Closed-Loop Control on Programmable Experimental Platforms,” IEEE
Transactions on Mobile Computing, vol. 22, no. 10, pp. 5787–5800,
2023.
```
[36] O-RAN, “O-RAN Working Group 2 (A1 interface: Type Definitions) (O-
```
```
RAN.WG2.A1TD-R003-v08.00),” Available at https://www.o-ran.org/
```
```
(2025/01/20), O-RAN Alliance, Tech. Rep., 2024.
```
[37] S. D’Oro, M. Polese, L. Bonati, H. Cheng, and T. Melodia, “dApps:
Distributed Applications for Real-Time Inference and Control in O-
RAN,” IEEE Communications Magazine, vol. 60, no. 11, pp. 52–58,
2022.
[38] N. M. Yungaicela-Naula, V. Sharma, and S. Scott-Hayward, “Miscon-
figuration in O-RAN: Analysis of the impact of AI/ML,” Computer
Networks, vol. 247, p. 110455, 2024.
Noe M. Yungaicela-Naula received his Ph.D. degree in Engineering Sciences
from Tecnologico de Monterrey, Mexico, in 2023. He joined the Centre
```
for Secure Information Technologies (CSIT) at Queen’s University Belfast
```
```
(QUB), UK, as a Research Fellow in 2023. His current research interests
```
include the use of AI techniques to automate different tasks in next-generation
```
networks (SDN, NFV, 5G, Open RAN, and ICS), such as intrusion detection
```
systems, anomaly detection, and network resource optimization.
```
Vishal Sharma (Senior Member, IEEE) is a Reader at the School of
```
```
Electronics, Electrical Engineering and Computer Science (EEECS), Queen’s
```
University Belfast, Northern Ireland, United Kingdom. He is the director of
the Innovation-by-design lab at QUB. Previously, he led the British Computer
```
Society (BCS) Accreditation for QUB and was the Chair of the Computer
```
Science Programme Review Working Group. He is the Northern Ireland
```
Advanced Research and Engineering Centre (ARC) co-investigator. He also
```
served as the co-chair of the IEEE UK and Ireland Diversity, Equity, and
Inclusion Committee. Before coming to QUB, he was a Research Fellow
in the Information Systems Technology and Design Pillar at the Singapore
University of Technology and Design, where he worked on the future-proof
blockchain systems funded by MoE, Singapore. From November 2016 to
March 2019, he worked in multiple positions in the Information Security
Engineering Department at Soonchunhyang University, South Korea. Before
this, he worked as a lecturer in the Computer Science and Engineering
Department at Thapar University, India. He has authored/co-authored more
than 150 journal/conference articles and book chapters, co-edited four books,
and received nine best paper awards. His authored work is included in the
IEEE ComSoc’s best readings in UAV Assisted Wireless Networks, Topic:
Interference Mitigation and Drone Deployment. In 2024, he was recognised
with QUB’s Individual Performance Award. At present, Dr Sharma leads
research in Cyber Defence and Secure Computing, with an emphasis on
Network and Mobile Internet Security, Unmanned Aerial Systems, Agentic
AI, Distributed Ledger Technology, and Supply Chain Security. He is a senior
member of IEEE, a professional member of ACM, a Fellow of the Higher
Education Academy via the Queen’s Merit Award, and serves on the Royal
Society International Exchanges Committee.
```
Sandra Scott-Hayward (Senior Member, IEEE) is an Associate Professor
```
with the School of Electronics, Electrical Engineering and Computer Sci-
ence, and a Member of the Centre for Secure Information Technologies at
```
Queen’s University Belfast (QUB). She began her career in industry and
```
became a Chartered Engineer in 2006. Since joining academia, she has
contributed security designs and solutions for softwarized networks based
on her research on network security architectures and security functions for
emerging networks, specifically considering threat detection and protection
mechanisms in programmable networks. She received Outstanding Technical
Contributor and Outstanding Leadership awards from the Open Networking
Foundation in 2015 and 2016, respectively, having been elected and serving
as the Vice-Chair of the ONF Security Working Group from 2015 to 2017.
Dr. Scott-Hayward serves on the editorial board for IEEE Transactions
on Network and Service Management, and IEEE Transactions on Machine
Learning in Communications and Networking. She is Vice-Chair of the
IEEE NetSoft Steering Committee. She is Director of the QUB Academic
```
Centre of Excellence in Cyber Security Education (ACE-CSE), co-lead of
```
the QUB Leverhulme Interdisciplinary Network on Algorithmic Solutions
```
(LINAS) doctoral training programme and was a Polymath Fellow of the
```
```
Global Fellowship Initiative at the Geneva Centre for Security Policy (GCSP)
```
from 2021 to 2023. With LINAS and GCSP, she explores the impact of ML
and AI technologies on security and society.
This article has been accepted for publication in IEEE Transactions on Mobile Computing. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TMC.2026.3666787
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: UNIVERSIDADE FEDERAL DO PARA. Downloaded on February 26,2026 at 18:13:50 UTC from IEEE Xplore. Restrictions apply.