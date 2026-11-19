python3 tools/hw_topo.py

================================================================================================
 System overview
================================================================================================
 Host aquila   Kernel 6.8.0-147-generic   2026-11-11 09:20
 CPU  Intel(R) Xeon(R) 6740P  × 2 socket, 4 NUMA nodes
 NUMA 0: CPU 0-23,96-119          Memory 504 GB   GPU: GPU0, GPU1   NIC: NIC0, NIC1, NIC2, NIC3   Accel: QAT×1 DSA×1 IAA×1
 NUMA 1: CPU 24-47,120-143        Memory 504 GB   GPU: GPU2, GPU3   NIC: NIC4, NIC5   Accel: QAT×1 DSA×1 IAA×1
 NUMA 2: CPU 48-71,144-167        Memory 504 GB   GPU: GPU4, GPU5   NIC: NIC6, NIC7, NIC8, NIC9, NIC10, NIC11, NIC12, NIC13   Accel: QAT×1 DSA×1 IAA×1
 NUMA 3: CPU 72-95,168-191        Memory 504 GB   GPU: GPU6, GPU7   NIC: NIC14, NIC15   Accel: QAT×1 DSA×1 IAA×1
 IOMMU on (188 groups)   Kernel args [intel_iommu=on,sm_on iommu=pt]   ACS config needs root to read

================================================================================================
 GPUs (index matches nvidia-smi; Link = current/max; see the HTML page for full details)
================================================================================================
 GPU   PCI addr      Name [vendor:dev]                NUMA  Local CPUs     Link             BAR1     Memory   Upstream
 ----  ------------  -------------------------------  ----  -------------  ---------------  -------  -------  ----------
 GPU0  0000:10:00.0  NVIDIA B300 SXM6 AC [10de:3182]  0     0-23,96-119    Gen6x16/Gen6x16  512 GiB  269 GiB  SW 0e:00.0
 GPU1  0000:38:00.0  NVIDIA B300 SXM6 AC [10de:3182]  0     0-23,96-119    Gen6x16/Gen6x16  512 GiB  269 GiB  SW 36:00.0
 GPU2  0000:5e:00.0  NVIDIA B300 SXM6 AC [10de:3182]  1     24-47,120-143  Gen6x16/Gen6x16  512 GiB  269 GiB  SW 5c:00.0
 GPU3  0000:71:00.0  NVIDIA B300 SXM6 AC [10de:3182]  1     24-47,120-143  Gen6x16/Gen6x16  512 GiB  269 GiB  SW 6f:00.0
 GPU4  0000:90:00.0  NVIDIA B300 SXM6 AC [10de:3182]  2     48-71,144-167  Gen6x16/Gen6x16  512 GiB  269 GiB  SW 8e:00.0
 GPU5  0000:b9:00.0  NVIDIA B300 SXM6 AC [10de:3182]  2     48-71,144-167  Gen6x16/Gen6x16  512 GiB  269 GiB  SW b7:00.0
 GPU6  0000:dd:00.0  NVIDIA B300 SXM6 AC [10de:3182]  3     72-95,168-191  Gen6x16/Gen6x16  512 GiB  269 GiB  SW db:00.0
 GPU7  0000:ef:00.0  NVIDIA B300 SXM6 AC [10de:3182]  3     72-95,168-191  Gen6x16/Gen6x16  512 GiB  269 GiB  SW ed:00.0

================================================================================================
 NICs
================================================================================================
 NIC     PCI addr      Name [vendor:dev]                         Netdev/RDMA state                             NUMA  Link             Upstream
 ------  ------------  ----------------------------------------  --------------------------------------------  ----  ---------------  ----------
 NIC0*   0000:0d:00.0  Mellanox Technologies CX8 Family [Connec  ibp13s0 down; mlx5_0/p1 DOWN 10G SDR          0     Gen6x16/Gen6x16  SW 0b:00.0
 NIC1*   0000:35:00.0  Mellanox Technologies CX8 Family [Connec  ibp53s0 down; mlx5_1/p1 DOWN 10G SDR          0     Gen6x16/Gen6x16  SW 33:00.0
 NIC2*   0000:48:00.0  Mellanox Technologies MT2892 Family [Con  ens1f0np0 down; mlx5_2/p1 DOWN 40G QDR        0     Gen4x16/Gen4x16  SW 46:00.0
 NIC3*   0000:48:00.1  Mellanox Technologies MT2892 Family [Con  ens1f1np1 up 100G; mlx5_3/p1 ACTIVE 100G EDR  0     Gen4x16/Gen4x16  SW 46:00.0
 NIC4*   0000:5b:00.0  Mellanox Technologies CX8 Family [Connec  ibp91s0 down; mlx5_4/p1 DOWN 10G SDR          1     Gen6x16/Gen6x16  SW 59:00.0
 NIC5*   0000:6e:00.0  Mellanox Technologies CX8 Family [Connec  ibp110s0 down; mlx5_5/p1 DOWN 10G SDR         1     Gen6x16/Gen6x16  SW 6c:00.0
 NIC6*   0000:8d:00.0  Mellanox Technologies CX8 Family [Connec  ibp141s0 down; mlx5_6/p1 DOWN 10G SDR         2     Gen6x16/Gen6x16  SW 8b:00.0
 NIC7    0000:a2:00.0  Intel Corporation Ethernet Controller X7  ens3923f0np0 down                             2     Gen3x4/Gen3x4    RP a1:06.0
 NIC8    0000:a2:00.1  Intel Corporation Ethernet Controller X7  ens3923f1np1 down                             2     Gen3x4/Gen3x4    RP a1:06.0
 NIC9*   0000:a3:00.0  Mellanox Technologies MT2910 Family [Con  ibs3921f0 down; mlx5_7/p1 ACTIVE 100G HDR     2     Gen4x2/Gen4x2    RP a1:08.0
 NIC10*  0000:a3:00.1  Mellanox Technologies MT2910 Family [Con  ibs3921f1 down; mlx5_8/p1 ACTIVE 100G HDR     2     Gen4x2/Gen4x2    RP a1:08.0
 NIC11*  0000:a3:00.2  Mellanox Technologies MT2910 Family [Con  ibs3921f2 down; mlx5_9/p1 ACTIVE 100G HDR     2     Gen4x2/Gen4x2    RP a1:08.0
 NIC12*  0000:a3:00.3  Mellanox Technologies MT2910 Family [Con  ibs3921f3 down; mlx5_10/p1 ACTIVE 100G HDR    2     Gen4x2/Gen4x2    RP a1:08.0
 NIC13*  0000:b6:00.0  Mellanox Technologies CX8 Family [Connec  ibp182s0 down; mlx5_11/p1 DOWN 10G SDR        2     Gen6x16/Gen6x16  SW b4:00.0
 NIC14*  0000:da:00.0  Mellanox Technologies CX8 Family [Connec  ibp218s0 down; mlx5_12/p1 DOWN 10G SDR        3     Gen6x16/Gen6x16  SW d8:00.0
 NIC15*  0000:ec:00.0  Mellanox Technologies CX8 Family [Connec  ibp236s0 down; mlx5_13/p1 DOWN 10G SDR        3     Gen6x16/Gen6x16  SW ea:00.0
 * = Mellanox/NVIDIA NIC (supports GPUDirect RDMA)

================================================================================================
 Accelerators (Intel QAT / DSA / IAA; on-die = Root Complex integrated, no PCIe link; DSA/IAA tags = idxd names)
================================================================================================
 Accel  PCI addr      Name [vendor:dev]           NUMA  Driver  Location  State                                      
 -----  ------------  --------------------------  ----  ------  --------  --------------------------------------------
 QAT0   0000:01:00.0  Intel QAT 4xxx [8086:4944]  0     4xxx    on-die    VFs 0/16                                   
 QAT1   0000:06:00.0  Intel QAT 4xxx [8086:4944]  1     4xxx    on-die    VFs 0/16                                   
 QAT2   0000:81:00.0  Intel QAT 4xxx [8086:4944]  2     4xxx    on-die    VFs 0/16                                   
 QAT3   0000:86:00.0  Intel QAT 4xxx [8086:4944]  3     4xxx    on-die    VFs 0/16                                   
 dsa0   0000:00:01.0  Intel DSA [8086:0b25]       0     idxd    on-die    enabled, WQ 1/16 enabled (1 user), 4 engines
 dsa2   0000:05:01.0  Intel DSA [8086:0b25]       1     idxd    on-die    enabled, WQ 1/16 enabled (1 user), 4 engines
 dsa4   0000:80:01.0  Intel DSA [8086:0b25]       2     idxd    on-die    enabled, WQ 1/16 enabled (1 user), 4 engines
 dsa6   0000:85:01.0  Intel DSA [8086:0b25]       3     idxd    on-die    enabled, WQ 1/16 enabled (1 user), 4 engines
 iax1   0000:00:02.0  Intel IAA [8086:0cfe]       0     idxd    on-die    enabled, WQ 1/16 enabled (1 user), 8 engines
 iax3   0000:05:02.0  Intel IAA [8086:0cfe]       1     idxd    on-die    enabled, WQ 1/16 enabled (1 user), 8 engines
 iax5   0000:80:02.0  Intel IAA [8086:0cfe]       2     idxd    on-die    enabled, WQ 1/16 enabled (1 user), 8 engines
 iax7   0000:85:02.0  Intel IAA [8086:0cfe]       3     idxd    on-die    enabled, WQ 1/16 enabled (1 user), 8 engines

================================================================================================
 Topology tree (NUMA → Root Port → PCIe Switch → device; bandwidth = theoretical per direction)
================================================================================================
 NUMA 0  CPU 0-23,96-119
 ├─ on-die QAT ×1: QAT0 01:00.0
 ├─ on-die DSA ×1: dsa0 00:01.0
 ├─ on-die IAA ×1: iax1 00:02.0
 ├─ RootPort 0000:0a:02.0  [pci0000:0a]
 │  └─ PCIe Switch 0000:0b:00.0  Mellanox Technologies ConnectX/BlueField ...  ── uplink Gen5x16 63.0 GB/s
 │     │  downstream GPU0, NIC0 total 242.0 GB/s → uplink oversubscribed 3.8:1  (shared under concurrent traffic)
 │     ├─ port 0c:00.0 ── Gen6x16 121.0 GB/s ── NIC0 0000:0d:00.0  Mellanox Technologies CX8 Family [Con...  [ibp13s0 down; mlx5_0/p1 DOWN 10G SDR]
 │     └─ port 0c:02.0
 │        └─ PCIe Switch 0000:0e:00.0  Mellanox Technologies ConnectX/BlueField ...  ── uplink Gen6x16 121.0 GB/s
 │           │  downstream GPU0 total 121.0 GB/s → uplink oversubscribed 1.0:1
 │           └─ port 0f:00.0 ── Gen6x16 121.0 GB/s ── GPU0 0000:10:00.0  NVIDIA B300 SXM6 AC
 ├─ RootPort 0000:32:02.0  [pci0000:32]
 │  └─ PCIe Switch 0000:33:00.0  Mellanox Technologies ConnectX/BlueField ...  ── uplink Gen5x16 63.0 GB/s
 │     │  downstream GPU1, NIC1 total 242.0 GB/s → uplink oversubscribed 3.8:1  (shared under concurrent traffic)
 │     ├─ port 34:00.0 ── Gen6x16 121.0 GB/s ── NIC1 0000:35:00.0  Mellanox Technologies CX8 Family [Con...  [ibp53s0 down; mlx5_1/p1 DOWN 10G SDR]
 │     └─ port 34:02.0
 │        └─ PCIe Switch 0000:36:00.0  Mellanox Technologies ConnectX/BlueField ...  ── uplink Gen6x16 121.0 GB/s
 │           │  downstream GPU1 total 121.0 GB/s → uplink oversubscribed 1.0:1
 │           └─ port 37:00.0 ── Gen6x16 121.0 GB/s ── GPU1 0000:38:00.0  NVIDIA B300 SXM6 AC
 └─ RootPort 0000:45:02.0  [pci0000:45]
    └─ PCIe Switch 0000:46:00.0  Broadcom / LSI PEX890xx PCIe Gen 5 Switch  ── uplink Gen5x16 63.0 GB/s
       │  downstream NIC2, NIC3 total 31.5 GB/s → uplink oversubscribed 0.5:1
       ├─ port 47:00.0
       │  ├─ NIC2 0000:48:00.0  Mellanox Technologies MT2892 Family [...  [ens1f0np0 down; mlx5_2/p1 DOWN 40G QDR] ── Gen4x16 31.5 GB/s
       │  └─ NIC3 0000:48:00.1  Mellanox Technologies MT2892 Family [...  [ens1f1np1 up 100G; mlx5_3/p1 ACTIVE 100G EDR] ── Gen4x16 31.5 GB/s
       ├─ other ×2: 49:00.0 (Solidigm NVMe DC SSD [Atomos Prime]), 4e:00.0 (Broadcom / LSI PCIe
       │  Switch manage...)
       └─ empty ports ×4: 47:02.0, 47:03.0, 47:04.0, 47:05.0

 NUMA 1  CPU 24-47,120-143
 ├─ on-die QAT ×1: QAT1 06:00.0
 ├─ on-die DSA ×1: dsa2 05:01.0
 ├─ on-die IAA ×1: iax3 05:02.0
 ├─ RootPort 0000:58:02.0  [pci0000:58]
 │  └─ PCIe Switch 0000:59:00.0  Mellanox Technologies ConnectX/BlueField ...  ── uplink Gen5x16 63.0 GB/s
 │     │  downstream GPU2, NIC4 total 242.0 GB/s → uplink oversubscribed 3.8:1  (shared under concurrent traffic)
 │     ├─ port 5a:00.0 ── Gen6x16 121.0 GB/s ── NIC4 0000:5b:00.0  Mellanox Technologies CX8 Family [Con...  [ibp91s0 down; mlx5_4/p1 DOWN 10G SDR]
 │     └─ port 5a:02.0
 │        └─ PCIe Switch 0000:5c:00.0  Mellanox Technologies ConnectX/BlueField ...  ── uplink Gen6x16 121.0 GB/s
 │           │  downstream GPU2 total 121.0 GB/s → uplink oversubscribed 1.0:1
 │           └─ port 5d:00.0 ── Gen6x16 121.0 GB/s ── GPU2 0000:5e:00.0  NVIDIA B300 SXM6 AC
 └─ RootPort 0000:6b:02.0  [pci0000:6b]
    └─ PCIe Switch 0000:6c:00.0  Mellanox Technologies ConnectX/BlueField ...  ── uplink Gen5x16 63.0 GB/s
       │  downstream GPU3, NIC5 total 242.0 GB/s → uplink oversubscribed 3.8:1  (shared under concurrent traffic)
       ├─ port 6d:00.0 ── Gen6x16 121.0 GB/s ── NIC5 0000:6e:00.0  Mellanox Technologies CX8 Family [Con...  [ibp110s0 down; mlx5_5/p1 DOWN 10G SDR]
       └─ port 6d:02.0
          └─ PCIe Switch 0000:6f:00.0  Mellanox Technologies ConnectX/BlueField ...  ── uplink Gen6x16 121.0 GB/s
             │  downstream GPU3 total 121.0 GB/s → uplink oversubscribed 1.0:1
             └─ port 70:00.0 ── Gen6x16 121.0 GB/s ── GPU3 0000:71:00.0  NVIDIA B300 SXM6 AC

 NUMA 2  CPU 48-71,144-167
 ├─ on-die QAT ×1: QAT2 81:00.0
 ├─ on-die DSA ×1: dsa4 80:01.0
 ├─ on-die IAA ×1: iax5 80:02.0
 ├─ RootPort 0000:8a:02.0  [pci0000:8a]
 │  └─ PCIe Switch 0000:8b:00.0  Mellanox Technologies ConnectX/BlueField ...  ── uplink Gen5x16 63.0 GB/s
 │     │  downstream GPU4, NIC6 total 242.0 GB/s → uplink oversubscribed 3.8:1  (shared under concurrent traffic)
 │     ├─ port 8c:00.0 ── Gen6x16 121.0 GB/s ── NIC6 0000:8d:00.0  Mellanox Technologies CX8 Family [Con...  [ibp141s0 down; mlx5_6/p1 DOWN 10G SDR]
 │     └─ port 8c:02.0
 │        └─ PCIe Switch 0000:8e:00.0  Mellanox Technologies ConnectX/BlueField ...  ── uplink Gen6x16 121.0 GB/s
 │           │  downstream GPU4 total 121.0 GB/s → uplink oversubscribed 1.0:1
 │           └─ port 8f:00.0 ── Gen6x16 121.0 GB/s ── GPU4 0000:90:00.0  NVIDIA B300 SXM6 AC
 ├─ RootPort 0000:a1:06.0  [pci0000:a1]
 │  ├─ NIC7 0000:a2:00.0  Intel Corporation Ethernet Controller...  [ens3923f0np0 down] ── Gen3x4 3.9 GB/s
 │  └─ NIC8 0000:a2:00.1  Intel Corporation Ethernet Controller...  [ens3923f1np1 down] ── Gen3x4 3.9 GB/s
 ├─ RootPort 0000:a1:08.0  [pci0000:a1]
 │  ├─ NIC9 0000:a3:00.0  Mellanox Technologies MT2910 Family [...  [ibs3921f0 down; mlx5_7/p1 ACTIVE 100G HDR] ── Gen4x2 3.9 GB/s
 │  ├─ NIC10 0000:a3:00.1  Mellanox Technologies MT2910 Family [...  [ibs3921f1 down; mlx5_8/p1 ACTIVE 100G HDR] ── Gen4x2 3.9 GB/s
 │  ├─ NIC11 0000:a3:00.2  Mellanox Technologies MT2910 Family [...  [ibs3921f2 down; mlx5_9/p1 ACTIVE 100G HDR] ── Gen4x2 3.9 GB/s
 │  └─ NIC12 0000:a3:00.3  Mellanox Technologies MT2910 Family [...  [ibs3921f3 down; mlx5_10/p1 ACTIVE 100G HDR] ── Gen4x2 3.9 GB/s
 └─ RootPort 0000:b3:02.0  [pci0000:b3]
    └─ PCIe Switch 0000:b4:00.0  Mellanox Technologies ConnectX/BlueField ...  ── uplink Gen5x16 63.0 GB/s
       │  downstream GPU5, NIC13 total 242.0 GB/s → uplink oversubscribed 3.8:1  (shared under concurrent traffic)
       ├─ port b5:00.0 ── Gen6x16 121.0 GB/s ── NIC13 0000:b6:00.0  Mellanox Technologies CX8 Family [Con...  [ibp182s0 down; mlx5_11/p1 DOWN 10G SDR]
       └─ port b5:02.0
          └─ PCIe Switch 0000:b7:00.0  Mellanox Technologies ConnectX/BlueField ...  ── uplink Gen6x16 121.0 GB/s
             │  downstream GPU5 total 121.0 GB/s → uplink oversubscribed 1.0:1
             └─ port b8:00.0 ── Gen6x16 121.0 GB/s ── GPU5 0000:b9:00.0  NVIDIA B300 SXM6 AC

 NUMA 3  CPU 72-95,168-191
 ├─ on-die QAT ×1: QAT3 86:00.0
 ├─ on-die DSA ×1: dsa6 85:01.0
 ├─ on-die IAA ×1: iax7 85:02.0
 ├─ RootPort 0000:d7:02.0  [pci0000:d7]
 │  └─ PCIe Switch 0000:d8:00.0  Mellanox Technologies ConnectX/BlueField ...  ── uplink Gen5x16 63.0 GB/s
 │     │  downstream GPU6, NIC14 total 242.0 GB/s → uplink oversubscribed 3.8:1  (shared under concurrent traffic)
 │     ├─ port d9:00.0 ── Gen6x16 121.0 GB/s ── NIC14 0000:da:00.0  Mellanox Technologies CX8 Family [Con...  [ibp218s0 down; mlx5_12/p1 DOWN 10G SDR]
 │     └─ port d9:02.0
 │        └─ PCIe Switch 0000:db:00.0  Mellanox Technologies ConnectX/BlueField ...  ── uplink Gen6x16 121.0 GB/s
 │           │  downstream GPU6 total 121.0 GB/s → uplink oversubscribed 1.0:1
 │           └─ port dc:00.0 ── Gen6x16 121.0 GB/s ── GPU6 0000:dd:00.0  NVIDIA B300 SXM6 AC
 └─ RootPort 0000:e9:02.0  [pci0000:e9]
    └─ PCIe Switch 0000:ea:00.0  Mellanox Technologies ConnectX/BlueField ...  ── uplink Gen5x16 63.0 GB/s
       │  downstream GPU7, NIC15 total 242.0 GB/s → uplink oversubscribed 3.8:1  (shared under concurrent traffic)
       ├─ port eb:00.0 ── Gen6x16 121.0 GB/s ── NIC15 0000:ec:00.0  Mellanox Technologies CX8 Family [Con...  [ibp236s0 down; mlx5_13/p1 DOWN 10G SDR]
       └─ port eb:02.0
          └─ PCIe Switch 0000:ed:00.0  Mellanox Technologies ConnectX/BlueField ...  ── uplink Gen6x16 121.0 GB/s
             │  downstream GPU7 total 121.0 GB/s → uplink oversubscribed 1.0:1
             └─ port ee:00.0 ── Gen6x16 121.0 GB/s ── GPU7 0000:ef:00.0  NVIDIA B300 SXM6 AC


================================================================================================
 GPU ↔ GPU affinity matrix (same as nvidia-smi topo -m)
================================================================================================
          GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7
 GPU0        X  NODE   SYS   SYS   SYS   SYS   SYS   SYS
 GPU1     NODE     X   SYS   SYS   SYS   SYS   SYS   SYS
 GPU2      SYS   SYS     X  NODE   SYS   SYS   SYS   SYS
 GPU3      SYS   SYS  NODE     X   SYS   SYS   SYS   SYS
 GPU4      SYS   SYS   SYS   SYS     X  NODE   SYS   SYS
 GPU5      SYS   SYS   SYS   SYS  NODE     X   SYS   SYS
 GPU6      SYS   SYS   SYS   SYS   SYS   SYS     X  NODE
 GPU7      SYS   SYS   SYS   SYS   SYS   SYS  NODE     X

   PIX   same PCIe switch, no CPU involved
   PXB   multiple PCIe switches, no CPU involved
   PHB   same host bridge, via CPU root complex
   NODE  same NUMA node, different host bridge, via CPU
   SYS   different NUMA nodes, via UPI

================================================================================================
 GPU ↔ NIC affinity matrix (prefer PIX/PXB for GPUDirect RDMA)
================================================================================================
           NIC0   NIC1   NIC2   NIC3   NIC4   NIC5   NIC6   NIC7   NIC8   NIC9  NIC10  NIC11  NIC12  NIC13  NIC14  NIC15
 GPU0       PXB   NODE   NODE   NODE    SYS    SYS    SYS    SYS    SYS    SYS    SYS    SYS    SYS    SYS    SYS    SYS
 GPU1      NODE    PXB   NODE   NODE    SYS    SYS    SYS    SYS    SYS    SYS    SYS    SYS    SYS    SYS    SYS    SYS
 GPU2       SYS    SYS    SYS    SYS    PXB   NODE    SYS    SYS    SYS    SYS    SYS    SYS    SYS    SYS    SYS    SYS
 GPU3       SYS    SYS    SYS    SYS   NODE    PXB    SYS    SYS    SYS    SYS    SYS    SYS    SYS    SYS    SYS    SYS
 GPU4       SYS    SYS    SYS    SYS    SYS    SYS    PXB   NODE   NODE   NODE   NODE   NODE   NODE   NODE    SYS    SYS
 GPU5       SYS    SYS    SYS    SYS    SYS    SYS   NODE   NODE   NODE   NODE   NODE   NODE   NODE    PXB    SYS    SYS
 GPU6       SYS    SYS    SYS    SYS    SYS    SYS    SYS    SYS    SYS    SYS    SYS    SYS    SYS    SYS    PXB   NODE
 GPU7       SYS    SYS    SYS    SYS    SYS    SYS    SYS    SYS    SYS    SYS    SYS    SYS    SYS    SYS   NODE    PXB

   NIC0 (Mellanox Technologies CX8 Family): nearest GPU GPU0 [PXB]
   NIC1 (Mellanox Technologies CX8 Family): nearest GPU GPU1 [PXB]
   NIC2 (Mellanox Technologies MT2892 Fam): nearest GPU GPU0, GPU1 [NODE]
   NIC3 (Mellanox Technologies MT2892 Fam): nearest GPU GPU0, GPU1 [NODE]
   NIC4 (Mellanox Technologies CX8 Family): nearest GPU GPU2 [PXB]
   NIC5 (Mellanox Technologies CX8 Family): nearest GPU GPU3 [PXB]
   NIC6 (Mellanox Technologies CX8 Family): nearest GPU GPU4 [PXB]
   NIC7 (Intel Corporation Ethernet Contr): nearest GPU GPU4, GPU5 [NODE]
   NIC8 (Intel Corporation Ethernet Contr): nearest GPU GPU4, GPU5 [NODE]
   NIC9 (Mellanox Technologies MT2910 Fam): nearest GPU GPU4, GPU5 [NODE]
   NIC10 (Mellanox Technologies MT2910 Fam): nearest GPU GPU4, GPU5 [NODE]
   NIC11 (Mellanox Technologies MT2910 Fam): nearest GPU GPU4, GPU5 [NODE]
   NIC12 (Mellanox Technologies MT2910 Fam): nearest GPU GPU4, GPU5 [NODE]
   NIC13 (Mellanox Technologies CX8 Family): nearest GPU GPU5 [PXB]
   NIC14 (Mellanox Technologies CX8 Family): nearest GPU GPU6 [PXB]
   NIC15 (Mellanox Technologies CX8 Family): nearest GPU GPU7 [PXB]

================================================================================================
 Bandwidth sharing and P2P path summary
================================================================================================
 [Shared uplink] devices under the same switch share one link to the CPU:
   Switch 0000:0b:00.0 (NUMA 0): NIC0  uplink Gen5x16 63.0 GB/s, downstream 121.0 GB/s → 1.9:1  ⚠ oversubscribed
   Switch 0000:0e:00.0 (NUMA 0): GPU0  uplink Gen6x16 121.0 GB/s, downstream 121.0 GB/s → 1.0:1
   Switch 0000:33:00.0 (NUMA 0): NIC1  uplink Gen5x16 63.0 GB/s, downstream 121.0 GB/s → 1.9:1  ⚠ oversubscribed
   Switch 0000:36:00.0 (NUMA 0): GPU1  uplink Gen6x16 121.0 GB/s, downstream 121.0 GB/s → 1.0:1
   Switch 0000:46:00.0 (NUMA 0): NIC2, NIC3  uplink Gen5x16 63.0 GB/s, downstream 31.5 GB/s → 0.5:1
   Switch 0000:59:00.0 (NUMA 1): NIC4  uplink Gen5x16 63.0 GB/s, downstream 121.0 GB/s → 1.9:1  ⚠ oversubscribed
   Switch 0000:5c:00.0 (NUMA 1): GPU2  uplink Gen6x16 121.0 GB/s, downstream 121.0 GB/s → 1.0:1
   Switch 0000:6c:00.0 (NUMA 1): NIC5  uplink Gen5x16 63.0 GB/s, downstream 121.0 GB/s → 1.9:1  ⚠ oversubscribed
   Switch 0000:6f:00.0 (NUMA 1): GPU3  uplink Gen6x16 121.0 GB/s, downstream 121.0 GB/s → 1.0:1
   Switch 0000:8b:00.0 (NUMA 2): NIC6  uplink Gen5x16 63.0 GB/s, downstream 121.0 GB/s → 1.9:1  ⚠ oversubscribed
   Switch 0000:8e:00.0 (NUMA 2): GPU4  uplink Gen6x16 121.0 GB/s, downstream 121.0 GB/s → 1.0:1
   NIC7, NIC8: directly on RootPort 0000:a1:06.0, dedicated uplink
   NIC9, NIC10, NIC11, NIC12: directly on RootPort 0000:a1:08.0, dedicated uplink
   Switch 0000:b4:00.0 (NUMA 2): NIC13  uplink Gen5x16 63.0 GB/s, downstream 121.0 GB/s → 1.9:1  ⚠ oversubscribed
   Switch 0000:b7:00.0 (NUMA 2): GPU5  uplink Gen6x16 121.0 GB/s, downstream 121.0 GB/s → 1.0:1
   Switch 0000:d8:00.0 (NUMA 3): NIC14  uplink Gen5x16 63.0 GB/s, downstream 121.0 GB/s → 1.9:1  ⚠ oversubscribed
   Switch 0000:db:00.0 (NUMA 3): GPU6  uplink Gen6x16 121.0 GB/s, downstream 121.0 GB/s → 1.0:1
   Switch 0000:ea:00.0 (NUMA 3): NIC15  uplink Gen5x16 63.0 GB/s, downstream 121.0 GB/s → 1.9:1  ⚠ oversubscribed
   Switch 0000:ed:00.0 (NUMA 3): GPU7  uplink Gen6x16 121.0 GB/s, downstream 121.0 GB/s → 1.0:1
 [GPU P2P]
   Group A = {GPU0}  @ Switch 0000:0e:00.0, NUMA 0
   Group B = {GPU1}  @ Switch 0000:36:00.0, NUMA 0
   Group C = {GPU2}  @ Switch 0000:5c:00.0, NUMA 1
   Group D = {GPU3}  @ Switch 0000:6f:00.0, NUMA 1
   Group E = {GPU4}  @ Switch 0000:8e:00.0, NUMA 2
   Group F = {GPU5}  @ Switch 0000:b7:00.0, NUMA 2
   Group G = {GPU6}  @ Switch 0000:db:00.0, NUMA 3
   Group H = {GPU7}  @ Switch 0000:ed:00.0, NUMA 3
   Intra-group P2P stays inside the switch (PIX): no uplink or CPU involved;
   same NUMA, different group (NODE/PHB): forwarded by the CPU root complex, bounded by uplink and CPU;
   cross-NUMA (SYS): also crosses UPI, lowest bandwidth and highest latency.
 [ACS] run as root to check whether downstream-port ACS redirects P2P to the CPU (affects same-switch P2P)
 [Accel] on-die QAT / DSA / IAA have no PCIe link; use the ones on the GPU's NUMA node (tools/auto_config.sh does this):
   NUMA 0 (GPU: GPU0, GPU1)
     QAT: QAT0
     DSA: dsa0
     IAA: iax1
   NUMA 1 (GPU: GPU2, GPU3)
     QAT: QAT1
     DSA: dsa2
     IAA: iax3
   NUMA 2 (GPU: GPU4, GPU5)
     QAT: QAT2
     DSA: dsa4
     IAA: iax5
   NUMA 3 (GPU: GPU6, GPU7)
     QAT: QAT3
     DSA: dsa6
     IAA: iax7
