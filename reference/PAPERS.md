# Reference Papers

This index documents additional papers relevant to scaling, adaptation, and data generation for PFN/ICL tabular models.

## Papers

| arXiv ID   | Title                                                                                    | Why it matters for this repo                                                                                      | Local file                                                                         |
| ---------- | ---------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| —          | Accurate predictions on small data with a tabular foundation model (TabPFN v2)           | Core TabPFN architecture and synthetic prior design; primary source for PFN-based tabular learning.               | `reference/Accurate predictions on small data with a tabular foundation model.pdf` |
| 2502.17361 | A Closer Look at TabPFN v2: Understanding Its Strengths and Extending Its Capabilities   | Analysis of TabPFN v2 strengths/limitations; meta-feature insights and divide-and-conquer strategies for scaling. | `reference/A Closer Look at TabPFN v2.pdf`                                         |
| 2311.10609 | Scaling TabPFN: Sketching and Feature Selection for Tabular Prior-Data Fitted Networks   | Strategies for high-dimensional/large-table scalability via compression and feature selection.                    | `reference/Scaling_TabPFN_2311.10609.pdf`                                          |
| 2402.11137 | TuneTables: Context Optimization for Scalable Prior-Data Fitted Networks                 | Parameter-efficient adaptation of PFNs; useful for config presets and prior adaptation pathways.                  | `reference/TuneTables_2402.11137.pdf`                                              |
| 2406.05207 | Retrieval & Fine-Tuning for In-Context Tabular Models                                    | Retrieval-conditioned adaptation (LoCalPFN) for harder datasets; informs dataset-aware generation/evaluation.     | `reference/LoCalPFN_Retrieval_FineTuning_2406.05207.pdf`                           |
| 2406.05216 | TabPFGen -- Tabular Data Generation with TabPFN                                          | Directly about tabular data generation using PFN mechanisms.                                                      | `reference/TabPFGen_2406.05216.pdf`                                                |
| 2410.18164 | TabDPT: Scaling Tabular Foundation Models on Real Data                                   | Real-data pretraining and scaling behavior; supports realism/scaling design choices.                              | `reference/TabDPT_2410.18164.pdf`                                                  |
| 2411.10634 | Drift-Resilient TabPFN: In-Context Learning Temporal Distribution Shifts on Tabular Data | Drift-aware modeling; relevant for temporal/drift synthetic data presets.                                         | `reference/Drift_Resilient_TabPFN_2411.10634.pdf`                                  |
| 2502.02527 | TabPFN Unleashed: A Scalable and Effective Solution to Tabular Classification Problems   | Methods for bias/variance control and large-scale adaptation.                                                     | `reference/TabPFN_Unleashed_2502.02527.pdf`                                        |
| 2502.05564 | TabICL: A Tabular Foundation Model for In-Context Learning on Large Data                 | Predecessor architecture to TabICLv2 with scalability details relevant to implementation.                         | `reference/TabICL_2502.05564.pdf`                                                  |
| 2502.06684 | EquiTabPFN: A Target-Permutation Equivariant Prior Fitted Networks                       | Multi-target permutation equivariance ideas for robust label-space handling.                                      | `reference/EquiTabPFN_2502.06684.pdf`                                              |
| 2506.10914 | Foundation Models for Causal Inference via Prior-Data Fitted Networks                    | Causal PFN framing for future causal-data generation mode support.                                                | `reference/Foundation_Models_for_Causal_Inference_2506.10914.pdf`                  |
| 2602.11139 | TabICLv2: A better, faster, scalable, and open tabular foundation model                  | Primary reference; this repo implements the Appendix E synthetic data engine.                                     | `reference/TabICLv2.pdf`                                                           |

## Source URLs

- `https://doi.org/10.1038/s41586-024-08328-6` (TabPFN v2, Nature)
- `https://arxiv.org/abs/2311.10609`
- `https://arxiv.org/abs/2402.11137`
- `https://arxiv.org/abs/2406.05207`
- `https://arxiv.org/abs/2406.05216`
- `https://arxiv.org/abs/2410.18164`
- `https://arxiv.org/abs/2411.10634`
- `https://arxiv.org/abs/2502.02527`
- `https://arxiv.org/abs/2502.05564`
- `https://arxiv.org/abs/2502.06684`
- `https://arxiv.org/abs/2502.17361`
- `https://arxiv.org/abs/2506.10914`
- `https://arxiv.org/abs/2602.11139`
