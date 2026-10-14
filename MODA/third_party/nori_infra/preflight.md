# Preflight NORI/RIC — Fase 5.1 (infraestrutura E2)

> Data: 2026-10-14 · Agente: Engenheiro de infraestrutura RIC (implementador)
> Escopo: PLANO_IMPLEMENTACAO_SQR_NORI.md §3-Fase 5.1 + tarefa do orquestrador (5 passos).
> Regra respeitada: **nenhum pacote de sistema instalado/alterado** (nenhum `sudo apt`/`dpkg` executado); k8s/docker-daemon intocados (apenas builds/containers docker já autorizados pela tarefa).

## 1. Preflight do ambiente (comandos executados nesta sessão)

| Item | Estado | Evidência (comando → saída) |
|---|---|---|
| **docker** | ✅ OK | `docker ps` → daemon ativo (2 containers: ntopng, portainer); `docker --version` → `Docker version 29.8.2, build 7fc2dff` |
| **kubectl** | ❌ ausente | `which kubectl` → `comando não encontrado` |
| **kind** | ❌ ausente | `which kind` → `comando não encontrado` |
| **minikube** | ❌ ausente | `which minikube` → `comando não encontrado` |
| **gcc-11 / g++-11** | ❌ ausentes no host | `ls /usr/bin/gcc-11 /usr/bin/g++-11` → inexistentes; `gcc --version` → `gcc (Ubuntu 15.2.0-16ubuntu1) 15.2.0` |
| **libsctp-dev** | ✅ OK | `dpkg -l libsctp-dev` → `ii 1.0.21+dfsg-1build2 amd64`; `/usr/include/netinet/sctp.h` presente |
| **cmake** | ✅ OK | `cmake --version` → `cmake version 4.2.3` |
| **disco livre** | ✅ OK (54 GB) | `df -h /home/eliothluy` → `/dev/nvme0n1p1 228G 163G 54G 76% /` |
| **internet (github.com)** | ✅ OK | `git clone https://github.com/lasseufpa/e2sim.git` → sucesso; tarball `github.com/nokia/asn1c` baixado; build docker puxou de `gerrit.o-ran-sc.org` |
| VirtualBox / KVM | ❌ ausentes | `which virtualbox VBoxManage` → nada; `ls /dev/kvm` → inexistente |
| SO / recursos | — | Ubuntu 26.04.1 LTS, kernel 7.0.0-38, 16 CPUs, 15 GiB RAM (8,2 GiB disponíveis) |

## 2. Clones (terceiros, gitignored na raiz)

| Repo | Caminho | Commit | Observação |
|---|---|---|---|
| [lasseufpa/nori](https://github.com/lasseufpa/nori) | `MODA/third_party/nori/` | `9ea0db9` (= origin/main, verificado com `git fetch` + `git status -uno`) | Já existia no workspace; sincronizado hoje. Inclui `Dockerfile`, `docker-compose.yml`, `asn/v03` |
| [lasseufpa/e2sim](https://github.com/lasseufpa/e2sim) | `MODA/third_party/e2sim/` | `5173971` (clone novo, 14/10/2026) | HEAD = merge PR #1 (`e2sim-update`) |

`.gitignore` da raiz atualizado (linhas 11–13): `MODA/third_party/nori/`, `MODA/third_party/e2sim/`, `MODA/third_party/e2sim/*.deb` — mesmo padrão do clone existente (`MODA/ns-o-ran-gym/third_party/network-slicing/`). `git check-ignore` confirma os 3 caminhos ignorados; `nori_infra/` permanece rastreável.

## 3. Build do e2sim

### 3.1 Tentativa no host com gcc padrão (15.2.0) — **FALHOU** (2 causas, registradas)

Comando: `cd MODA/third_party/e2sim/e2sim && mkdir -p build && cd build && cmake .. -DDEV_PKG=1 -DLOG_LEVEL=3 && make package`
(equivalente do `build_e2sim.sh 3` **sem** os passos finais `sudo apt-get purge`/`sudo dpkg --install`, proibidos pela regra do workspace — o script original instala pacote de sistema).

- **Causa A (corrigida localmente):** runtime ASN.1 incompleto no fork — `constr_TYPE.h:51: fatal error: oer_decoder.h: Arquivo ou diretório inexistente`. O fork tem 671 arquivos gerados em `e2sim/src/ASN1c/` (specs `e2ap-v02.02.03.asn1`, `E2SM-KPM-R003-v03.00.asn1`, `E2SM-R003-v03.00.asn1`) mas **perdeu 54 skeletons do asn1c** (conjunto OER + tipos básicos). O gitlink `asn1c@9925dbbd` (raiz, sem `.gitmodules`) resolve no **nokia/asn1c** (verificado via API GitHub). Também falta `e2sim/src/ASN1c/CMakeLists.txt` (o CMakeLists raiz faz `add_subdirectory(src/ASN1c)` e referencia `asn1_objects`).
  - Fix local (dentro do clone gitignored): copiados os 54 skeletons ausentes do tarball `nokia/asn1c @ 9925dbbd` e replicado o `CMakeLists.txt` de `src/ASN1c` do upstream `wineslab/o-ran-e2sim@develop` (glob `*.c` → objeto `asn1_objects`), marcado com comentário de origem.
- **Causa B (fundamental, NÃO corrigida):** após o fix A, `src/encoding/encode_e2apv1.cpp:73` → `error: 'RICsubscriptionRequest_IEs_t' was not declared in this scope` (e dezenas mais: `E2setupRequestIEs_t`, `ProtocolIE_ID_id_RANfunctionsAdded`, …). A biblioteca do fork é escrita para **E2AP v1**, mas o ASN.1 regenerado é **E2AP v2.02.03** — incompatível. **Falharia também com gcc-11**: não é problema de compilador, é fork defasado.

> **Bloqueio formal solicitado pela tarefa (registro de comando):** caso se queira tentar o build nativo do fork com a toolchain recomendada, o comando de instalação é `sudo apt install gcc-11 g++-11 libsctp-dev` — **não executado por mim** (proibido; e não resolveria a causa B acima). `libsctp-dev` já está presente no host.

### 3.2 Caminho oficial do NORI (Docker, gcc-11) — ✅ **SUCESSO**

O `Dockerfile` do NORI (`MODA/third_party/nori/Dockerfile`) **não usa o fork lasseufpa/e2sim**: clona `gerrit.o-ran-sc.org/r/sim/e2-interface @ da6f82f2` (linhas 57–59), regenera o ASN.1 com o Nokia asn1c (linha 39) a partir das specs oficiais `asn/v03` do NORI (linha 64: **E2AP v3.01 + KPM v3.00 + RC v3.01**, flags `-fcompound-names -fno-include-deps -findirect-choice -no-gen-OER -no-gen-example`, linhas 68/75), aplica patch de conformidade X.691/10.6 em `constr_CHOICE.c` (linha 80) e patch de CMake (linha 84), e compila com **gcc-11** (linha 31, PPA `ubuntu-toolchain-r`).

Comando executado (estágio de build do e2sim, sem os estágios pesados de ns-3):

```
docker build --target e2sim-builder -t nori-e2sim-builder MODA/third_party/nori   # exit 0
```

Artefatos verificados dentro da imagem (`docker run --rm --entrypoint bash nori-e2sim-builder -c ...`):

| Artefato | Valor |
|---|---|
| Compilador | `gcc-11 (Ubuntu 11.5.0-10ubuntu1~20~ppa3) 11.5.0` |
| `libe2sim.a` | 13.108.008 bytes em `/tmp/e2-interface/e2sim/build/` |
| `e2sim-dev_1.0.0_amd64.deb` | 2.465.964 bytes |
| ASN.1 regenerado | 1118 arquivos `.c/.h` |

O `.deb` foi extraído para `MODA/third_party/e2sim/e2sim-dev_1.0.0_amd64.deb` (`docker create` + `docker cp`; gitignored). **Não instalei o pacote** (`sudo dpkg --install` do script original — proibido). Para instalá-lo no host, se o orquestrador/usuário desejar: `sudo dpkg --install MODA/third_party/e2sim/e2sim-dev_1.0.0_amd64.deb`.

### 3.3 Nota: instalação pré-existente de e2sim no host (não alterada)

`dpkg -l | grep e2sim` → `ii  e2sim-dev 1.0.0 amd64` instalado em **abril/2026** (anterior a este projeto — provavelmente do setup ns-O-RAN/wineslab anterior): `/usr/local/lib/libe2sim.a` (8,2 MB, 392 símbolos `T`) + 287 headers em `/usr/local/include/e2sim/` (linhagem com `E2setupRequest.h`, E2AP v1). Nada foi mexido. ⚠️ Ao linkar o NORI (5.2), **não** usar essa lib antiga — usar a `libe2sim.a`/`.deb` da §3.2 (E2AP v3.01).

## 4. RIC (Near-RT E2Term, O-RAN SC Release I) — ⛔ **BLOQUEADO** (pendente de ação do usuário)

Dois caminhos documentados, ambos dependem de software ausente:

1. **Kubernetes** (caminho do README do NORI — os exemplos pedem o IP do pod `deployment-ricplt-e2term-alpha-*`): `kubectl`, `kind` e `minikube` **ausentes**. Passos exatos para o usuário desbloquear:
   ```bash
   # ferramentas (escolher um runtime de cluster):
   sudo apt install -y kubectl        # ou binário: curl -LO "https://dl.k8s.io/release/$(curl -Ls https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
   # cluster local single-node:
   [ -f /usr/local/bin/kind ] || curl -Lo kind https://kind.sigs.k8s.io/dl/latest/kind-linux-amd64 && sudo install kind /usr/local/bin/kind && kind create cluster
   #   (alternativa: https://minikube.sigs.k8s.io/docs/start/)
   # depois: deploy do Near-RT RIC O-RAN SC Release I (namespace ricplt) seguindo os blueprints do wiki NORI
   #         https://github.com/lasseufpa/nori/wiki/Blueprint-NORIv2
   ```
   Recursos: 16 CPUs/15 GiB RAM/54 GB livres — suficientes para um cluster single-node leve.
2. **VM blueprint NORIv2** (caminho "mais leve" do wiki, pré-configurado com RIC+xApps): `virtualbox`/`VBoxManage` **ausentes** e `/dev/kvm` inexistente (KVM indisponível). Passos exatos:
   ```bash
   sudo apt install -y virtualbox    # não executado — bloqueio (instalação de sistema)
   # baixar blueprint: https://nextcloud.lasseufpa.org/s/jXaEsqECXiEQfys  (.zip → .vdi)
   # VirtualBox: nova VM (Linux/Oracle Linux), ≥8 GB RAM, ≥4 vCPU, "usar disco existente",
   #             rede "Bridged Adapter"; login openran-br / openran-br
   # dentro da VM: ~/xapp-nori/{redeploy_ric.sh, update_xapp.sh, resubscribe.sh}
   ```

Observação: o `docker-compose.yml` do NORI (`nori-dev`) sobe apenas o **ambiente de build/uso do ns-3.42+NR+NORI** (com e2sim já empacotado) — **não** sobe E2Term/RIC; não é caminho para o RIC. Estado registrado: **RIC PENDENTE** (não bloqueia a validação do e2sim — ver §5).

## 5. check_infra.sh (criado e executado)

`MODA/third_party/nori_infra/check_infra.sh` (executável, `chmod +x`). Verifica: docker funcional; artefato do e2sim (`.deb` docker/gcc-11 → `libe2sim.a` nativa → instalada); estado do RIC é **informativo** (pode ficar PENDENTE). Saída atual:

```
== check_infra (Fase 5.1 NORI) ==
docker..............: YES
e2sim artefato......: e2sim-dev .deb (docker/gcc-11): .../MODA/third_party/e2sim/e2sim-dev_1.0.0_amd64.deb (2,4M)
RIC (E2Term)........: PENDENTE (informativo)
INFRA OK
exit=0
```

`INFRA OK` só é impresso com docker + artefato do e2sim presentes — hoje, o artefato vem do build oficial §3.2.

## 6. Implicações para a Fase 5.2 (nota ao orquestrador/Agente A)

- **Não construir o port sobre o fork `lasseufpa/e2sim`** (ASN.1 v2.02.03 + biblioteca E2AP v1 — inconsistente). O caminho validado é `e2-interface @ da6f82f2` + regeneração ASN.1 das specs `asn/v03` do NORI, exatamente como o `Dockerfile` oficial (linhas 57–96).
- O NORI oficial ancora em **ns-3.42** (`ab4cce02`) + **NR 5G-LENA `78b7179e`** (linhas 92–97 do Dockerfile); o MODA tem ns-3.46-dev + NR v5.2-dev → o port de API permanece como previsto no plano (§2.4).
- Para linkar o módulo NORI no build host (gcc 15): usar a `libe2sim.a` §3.2 (gcc-11 é ABI-compatível com gcc 15 para libs estáticas C++; a parte C++ do e2sim é mínima). Alternativa robusta: compilar o NORI dentro do container `nori-dev` (`docker compose up -d`) — avaliar na 5.2.
- O `check_infra.sh` considera ok apenas e2sim; quando o RIC subir (k8s ou VM), ele passa a reportar o estado automaticamente (kubectl/VBoxManage detectados).

## 7. Resumo de bloqueios (para o usuário)

| # | Bloqueio | Ação do usuário | Desbloqueia |
|---|---|---|---|
| 1 | kubectl/kind/minikube ausentes (k8s RIC) | `sudo apt install -y kubectl` + cluster (kind/minikube) + deploy RIC Release I (wiki NORIv2) | Caminho k8s do RIC |
| 2 | VirtualBox ausente, `/dev/kvm` inexistente | `sudo apt install -y virtualbox` + baixar blueprint .vdi (Nextcloud LASSE) | Caminho VM do RIC |
| 3 | (Opcional) instalar o novo e2sim-dev no host | `sudo dpkg --install MODA/third_party/e2sim/e2sim-dev_1.0.0_amd64.deb` (substitui o de abril/2026) | Link direto do NORI no build host (senão, build dentro do container) |

— Nada além destes itens exigiu intervenção: e2sim **compilou com sucesso** (§3.2), artefato extraído e `INFRA OK`.
