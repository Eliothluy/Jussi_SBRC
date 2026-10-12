# Patches defensivos 5G-LENA (contrib/nr)

O módulo `contrib/nr` não é rastreado pelo git do projeto. Estas cópias contêm
os 3 patches defensivos aplicados sobre o 5G-LENA para que perdas pesadas em
RLC-UM não abortem a simulação (SIGABRT/NS_ASSERT) — sem eles, certas seeds
nos cenários congestion/insufficient_resources morrem em +2.4 s.

Instalação:
    cp nr-pdcp.cc       ../../ns-3-dev/contrib/nr/model/
    cp nr-rlc-um.cc     ../../ns-3-dev/contrib/nr/model/
    cp nr-net-device.cc ../../ns-3-dev/contrib/nr/model/
    cd ../../ns-3-dev && ./ns3 build rslaq-sim

Patches (defesa em profundidade, descartam PDU/SDU malformada com NS_LOG_WARN):
1. nr-pdcp.cc      — NrPdcp::DoReceivePdu: valida bit D/C e tamanho mínimo
                     antes de RemoveHeader (descarta PDU corrompida).
2. nr-rlc-um.cc    — NrRlcUm::ReassembleAndDeliver (estado WAITING_SI_SF sem
                     perda sinalizada): descarta PDU com Framing Info inesperado
                     em vez de NS_ASSERT_MSG fatal.
3. nr-net-device.cc— NrNetDevice::Receive: exige tamanho mínimo (20 B IPv4 /
                     40 B IPv6) antes de PeekHeader; SDU curta cai no branch
                     de drop já existente.

O 4º patch (renormalização de pesos no RslaqMacScheduler) está no arquivo
versionado ns-3-dev/scratch/rslaq/rslaq-mac-scheduler.cc.
