import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

# chroma_client = chromadb.PersistentClient("./chroma_db")
# chroma_collection = chroma_client.create_collection("quickstart")
# vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# chroma_collection = chroma_client.get_collection("quickstart")
# import pdb
# pdb.set_trace()
# print(chroma_collection)
# vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

p = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nAs a helpful assistant, you will utilize the provided document to answer user questions. \nRead the given document before providing answers and think step by step. \nThe document has an order of paragraphs with a higher correlation to the questions from the top to the bottom. \nThe answer may be hidden in the tables, so please find it as closely as possible. \nDo not use any other information to answer the user. Provide a etailed answer to the question.\nAlso, please provide the answer in the following order of priorities if applicable:\nFirstly, emphasize GPU characteristics and GPU products.\nSecondly, Give prominence to power-related specifications such as fan cooling or liquid cooling, power consumption, and so on.\nThirdly, If applicable, mention green computing.\nRemember, please don\'t provide any fabricated information, ensuring that everything stated is accurate and true.\n\nContext information is below.\n---------------------\nfile_path: /home/joyce/RAG-LlamaIndex/2.Pinecone -  HybridRetriever - Adv.Ingestion/data/sys-421gu-tnxr.pdf\n\nSupermicro Server Manager (SSM); Supermicro Power Manager (SPM); Supermicro Update Manager (SUM); SuperDoctor® 5; Super Diagnostics Offline; KVM with dedicated LAN; IPMI 2.0\n\n### PC Health Monitoring\n\n- CPU: Monitors for CPU Cores, Chipset Voltages, Memory\n- 8+4 Phase-switching voltage regulator\n- FAN: Fans with tachometer monitoring\n- Status monitor for speed control\n- Pulse Width Modulated (PWM) fan connectors\n- Temperature: Monitoring for CPU and chassis environment\n- Thermal Control for fan connectors\n\n### Dimensions and Weight\n\n|Height: 7" (175.6 mm)|Operating Temperature: 10°C ~ 35°C (50°F ~ 95°F)|\n|---|---|\n|Width: 17.67" (449 mm)|Non-operating Temperature: -40°C to 60°C (-40°F to 140°F)|\n|Depth: 32.79" (833 mm)|Operating Relative Humidity: 8% to 90% (non-condensing)|\n|Gross Weight: 140 lbs (64 kg)|Non-operating Relative Humidity: 5% to 95% (non-condensing)|\n|Net Weight: 105 lbs (48 kg)| |\n|Packaging: 14.57" (H) x 27.55" (W) x 49.6" (D)| |\n|Available Color: Black Front & Silver Body| |\n\n### Motherboard\n\nSuper X13DGU\n\n### Chassis\n\nCSE-458GTS-R3K06P\n\n©2024 Copyright Super Micro Computer, Inc. All rights reserved. *Product specifications may change without notice\n\nfile_path: /home/joyce/RAG-LlamaIndex/2.Pinecone -  HybridRetriever - Adv.Ingestion/data/sys-421gu-tnxr.pdf\n\n## SUPERMICR DATASHEET\n\nGPU SuperServer SYS-421GU-TNXR\n\nUniversal 4U Dual Processor (4th and 5th Gen Intel Scalable Processors) GPU System with NVIDIA HGX™ H100 4-GPU\n\nSXM5 board, NVLINK™ GPU-GPU Interconnect, and Redundant 3000W Titanium Level Power Supplies.\n\n### Key Applications\n\nHigh Performance Computing, AI/Deep Learning Training, Large Language Model (LLM) Natural Language Processing\n\n### Key Features\n\n- 32 DIMM slots Up to 8TB: 32x 256 GB DRAM Memory Type: 4800MHz ECC DDR5;\n- 32 DIMM slots Up to 8TB: 32x 256 GB DRAM Memory Type: 5600MTs ECC DDR5;\n- 8 PCIe Gen 5.0 X16 LP Slots;\n- Flexible networking options;\n- 2 M.2 NVMe and SATA for boot drive only; 6x 2.5" Hot-swap NVMe/SATA/SAS drive bays;\n\n### Form Factor\n\n4U Rackmount\nEnclosure: 449 x 175.6 x 833mm (17.67" x 7" x 32.79")\nPackage: 700 x 370 x 1260mm (27.55" x 14.57" x 49.6")\n\n### Processor\n\nDual Socket E (LGA-4677)\n5p Gen Intel® Xeon®/4p Gen Intel® Xeon® Scalable processors\nUp to 56C/112T; Up to 112.5MB Cache per CPU\n\n### GPU\n\nMax GPU Count: Up to 4 onboard GPU(s)\nSupported GPU: NVIDIA SXM: HGX H100 4-GPU (80GB)\nCPU-GPU Interconnect: PCIe 5.0 x16 CPU-to-GPU Interconnect\nGPU-GPU Interconnect: NVIDIA NVLink\n\n### System Memory\n\nSlot Count: 32 DIMM slots\nMax Memory (2DPC): Up to 8TB 5600MT/s ECC DDR5\n\n### Drive Bays\n\n6x 2.\n\nfile_path: /home/joyce/RAG-LlamaIndex/2.Pinecone -  HybridRetriever - Adv.Ingestion/data/sys-421gu-tnxr.pdf\n\nUp to 112.5MB Cache per CPU\n\n### GPU\n\nMax GPU Count: Up to 4 onboard GPU(s)\nSupported GPU: NVIDIA SXM: HGX H100 4-GPU (80GB)\nCPU-GPU Interconnect: PCIe 5.0 x16 CPU-to-GPU Interconnect\nGPU-GPU Interconnect: NVIDIA NVLink\n\n### System Memory\n\nSlot Count: 32 DIMM slots\nMax Memory (2DPC): Up to 8TB 5600MT/s ECC DDR5\n\n### Drive Bays\n\n6x 2.5" hot-swap NVMe/SATA drive bays (6x 2.5" NVMe hybrid)\n2 M.2 NVMe OR 2 M.2 SATA3\n\n### Expansion Slots\n\n1 PCIe 5.0 x16 LP slot(s)\n7 PCIe 5.0 X16 slot(s)\n\n### On-Board Devices\n\nChipset: Intel C741\nNetwork Connectivity: 2x 10GbE BaseT wip Intel X710-AT2\nIPMI: Support for Intelligent Platform Management Interface v.2.0 IPMI 2.0 wip virtual media over LAN and KVM-over-LAN support\n\n### Input / Output\n\nVideo: 1 VGA port(s)\n\n©2024 Copyright Super Micro Computer, Inc. All rights reserved. *Product specifications may change without notice\n---\n## DATASHEET\n\n|(Front View - System)|(Rear View - System)|\n|---|---|\n|Power Button|Reset Button|\n|Indicator LEDs|3000W Redundant Titanium Level Power Supplies|\n|Description Bay|Slot Description|\n|6-port Swappable NVMe SATA Drive|PCIe 20x6 Low Profile|\n\n### System Cooling\n\n- Fans: 5 heavy duty fans with optimal fan speed control\n- Air Shroud: 1 Air Shroud(s)\n- Liquid Cooling: 1 Direct to Chip (D2C) Cold Plate (optional)\n\n### Power Supply\n\n4x 3000W Redundant Titanium Level power supplies\n\n### System BIOS\n\nBIOS Type: AMI 32MB SPI Flash EEPROM\n\n### Management\n\nRedfish API; Supermicro Server Manager (SSM); Supermicro Power Manager (SPM);\n---------------------\nGiven the context information and not prior knowledge, answer the query.<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\nHow are you?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""
print(p)