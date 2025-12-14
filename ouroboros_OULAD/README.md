
# Ouroboros: Early identification of at-risk students Paper Replication

Reproducing paper: [Ouroboros: Early identification of at-risk students without models based on legacy data](https://oro.open.ac.uk/49731/1/paper.pdf)



## 🚀 Quick Start
0. Environment Setup

   ```
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
1. Download OULAD CSVs from the dataset page below.
2. Place CSVs into `ouroboros_OULAD_PaperReplicate/selflearner/data_load/data/` with these names:
   - `courses.csv`, `assessments.csv`, `vle.csv`, `studentInfo.csv`, `studentAssessment.csv`, `studentRegistration.csv`, `studentVle.csv`
3. Prepare the datasets, convert CSVs to HDF5:

```
python convert_csv_to_h5.py
```

This creates `selflearner/data_load/data/oulad.h5`

# Experiments

See [notebooks/ouroboros_experiments_new.ipynb](notebooks/ouroboros_experiments_new.ipynb)

Attention:
1. Follow the dataset split rules: splits are time-window based, not random.
2. Be careful with the label definition.
3. Describe the experiment/data section clearly.
4. Report and discuss trends in the results.
5. Feature construction can be slow; add a progress bar if needed.

## Results

### Table 4: PRAUC values for different days trained on the same presentation.

Paper:

| Day | SVM-W-R | SVM-R | LR | LR-W | NB | RF | XGB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.7790 | 0.7435 | 0.7561 | 0.7682 | 0.6779 | 0.7748 | 0.7442 |
| 1 | 0.6161 | 0.4081 | 0.5267 | 0.5944 | 0.4587 | 0.6184 | 0.5965 |
| 2 | 0.5436 | 0.3138 | 0.3852 | 0.4934 | 0.3673 | 0.5353 | 0.5315 |
| 3 | 0.4726 | 0.2629 | 0.3019 | 0.4164 | 0.3412 | 0.4960 | 0.5225 |
| 4 | 0.4596 | 0.2547 | 0.2866 | 0.3954 | 0.3577 | 0.4796 | 0.5079 |
| 5 | 0.4289 | 0.2363 | 0.2569 | 0.3870 | 0.3453 | 0.4600 | 0.4920 |
| 6 | 0.4171 | 0.2185 | 0.2195 | 0.3610 | 0.3475 | 0.4234 | 0.5200 |
| 7 | 0.4024 | 0.2027 | 0.2072 | 0.3263 | 0.3456 | 0.4309 | 0.4959 |
| 8 | 0.4118 | 0.1948 | 0.2272 | 0.3350 | 0.3487 | 0.4378 | 0.5309 |
| 9 | 0.3850 | 0.2031 | 0.2120 | 0.3260 | 0.3809 | 0.4820 | 0.5737 |
| 10 | 0.3677 | 0.2074 | 0.1967 | 0.3225 | 0.4011 | 0.4785 | 0.5669 |
| 11 | 0.3440 | 0.2033 | 0.1879 | 0.3039 | 0.3985 | 0.4569 | 0.5652 |

Replication:

| Day | SVM-W-R | SVM-R | LR | LR-W | NB | RF | XGB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.779198 | 0.750409 | 0.773795 | 0.774563 | 0.554130 | 0.777708 | 0.765853 |
| 1 | 0.620153 | 0.414209 | 0.614087 | 0.618889 | 0.387025 | 0.625210 | 0.576331 |
| 2 | 0.547465 | 0.321803 | 0.539832 | 0.535027 | 0.290650 | 0.549151 | 0.418607 |
| 3 | 0.478614 | 0.267980 | 0.484920 | 0.461042 | 0.266663 | 0.514068 | 0.374788 |
| 4 | 0.467145 | 0.252747 | 0.486392 | 0.440539 | 0.240311 | 0.495153 | 0.359332 |
| 5 | 0.436790 | 0.240853 | 0.464914 | 0.428756 | 0.219692 | 0.472223 | 0.310775 |
| 6 | 0.426464 | 0.217300 | 0.455717 | 0.411028 | 0.206099 | 0.447975 | 0.289286 |
| 7 | 0.409932 | 0.211095 | 0.450021 | 0.382155 | 0.185611 | 0.433190 | 0.292833 |
| 8 | 0.419791 | 0.208080 | 0.468553 | 0.381897 | 0.203484 | 0.456371 | 0.307548 |
| 9 | 0.416853 | 0.211446 | 0.501827 | 0.397380 | 0.204751 | 0.498681 | 0.273601 |
| 10 | 0.391519 | 0.220202 | 0.485158 | 0.376937 | 0.197703 | 0.478023 | 0.270392 |
| 11 | 0.368615 | 0.211847 | 0.479871 | 0.359718 | 0.190995 | 0.459813 | 0.254110 |



<!-- ### Figure 6: PR AUC for day 0 to 11 using Self-learning
<table>
  <tr>
    <td align="center"><strong>Paper</strong></td>
    <td align="center"><strong>Replication</strong></td>
  </tr>
  <tr>
    <td><img src="paper/Figure6_PRAUC_self_learning_paper.png" alt="Figure 6 — paper" width="100%"></td>
    <td><img src="paper/Figure6_PRAUC_self_learning_replicated.png" alt="Figure 6 — replicated" width="100%"></td>
  </tr>
</table>

### Figure 7: PR AUC for day 0 to 11 trained on legacy data
<table>
  <tr>
    <td align="center"><strong>Paper</strong></td>
    <td align="center"><strong>Replication</strong></td>
  </tr>
  <tr>
    <td><img src="paper/Figure7_PRAUC_legacy_paper.png" alt="Figure 7 — paper" width="100%"></td>
    <td><img src="paper/Figure7_PRAUC_legacy_replicated.png" alt="Figure 7 — replicated" width="100%"></td>
  </tr>
</table> -->

### Key Findings

#### Traditional ML Methods (Replication)
- Overall performance: replication results are now much closer to paper values, with day 0 performance nearly matching the paper (SVM-W-R: 0.7792 vs 0.7790, RF: 0.7777 vs 0.7748).
- Top performers: SVM-W-R and RF consistently lead across all days, with SVM-W-R achieving the highest day 0 PRAUC (0.7792). LR and LR-W also show strong performance, closely following the top models.
- Class weighting: LR-W slightly outperforms LR on day 0 (0.7746 vs 0.7738) but generally performs similarly or slightly worse on later days, suggesting limited benefit from class weighting in this replication.
- Trend over time: PRAUC decreases as prediction horizon extends, with most models showing gradual decline from day 0 to day 11. Notable exceptions include LR and RF showing slight recovery on day 9.
- Model-specific observations: NB remains the weakest performer throughout (0.19-0.55 range). XGB shows strong early performance (0.77 on day 0) but declines more rapidly than other top models, falling to 0.25 by day 11. SVM-R shows the largest gap compared to SVM-W-R, indicating the importance of class weighting for SVM.



## 📊 Dataset Description
- The original paper used HDF5 data, which is no longer available for download. The latest data is provided as CSV at https://analyse.kmi.open.ac.uk/open-dataset. You can use `convert_csv_to_h5.py` to convert CSV to HDF5 if needed.

For more dataset details, see the [dataset](https://analyse.kmi.open.ac.uk/open-dataset).
