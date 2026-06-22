********************************************************************************
* Program_R2.do
* JHSS revision (R2) — reviewer-requested specifications
*
* This file ASSUMES the analysis dataset has already been constructed by the
* original Program.do and saved as poll_final_data_with_change.dta. It does NOT
* re-scrape or re-classify; it runs only the additional / corrected analyses the
* reviewers requested, and writes the numbers reported in the revised manuscript.
*
* Each block is labeled with the reviewer comment it answers.
* Inferential choices throughout: WLS with aweight = sample1, standard errors
* clustered at the debate level [vce(cluster debate_date)], election-year FE i.year.
*
* IMPORTANT (Comments 13–14): incumbency, GDP (gdpbillionsofus) and per-capita GDP
* (percapitaus) take a single value per election year, so they are COLLINEAR with
* the election-year fixed effects and are intentionally NOT entered alongside i.year.
* The original draft's inclusion of both was the source of the reviewers' concern.
********************************************************************************

clear all
set more off
* ---- set this to the folder that holds the .dta ----
* cd "<path>/JHSS_2nd/R1/Data_Files"
use "poll_final_data_with_change.dta", clear

* estimation sample = polls with a valid pre-debate baseline and post-debate change
keep if !missing(change_democrats, change_republicans, pre_democrats, pre_republicans)
count
di "Estimation N (poll level); debate clusters follow:"
tab debate_date if e(sample)==0, missing
distinct debate_date

* convenience locals -------------------------------------------------------
local UNSIGNED  aggressive_words inflammatory_words discriminatory_words
local SRC       aggressive_words_d aggressive_words_r inflammatory_words_d inflammatory_words_r discriminatory_words_d discriminatory_words_r
local TOPIC     Immigration ForeignPolicy AbortionRights
local PRE       pre_democrats pre_republicans

********************************************************************************
* COMMENT 15 — Category prevalence by party (replaces implausible "0.0005%").
* Rhetoric is constant within candidate-debate, so describe at that unit.
********************************************************************************
preserve
    bysort debate_date candidate: keep if _n==1
    di "--- Democratic candidate-debates ---"
    tabstat `UNSIGNED' if democratdum==1, s(n mean sum) c(stat)
    count if inflammatory_words>0 & democratdum==1
    di "--- Republican candidate-debates ---"
    tabstat `UNSIGNED' if republicandum==1, s(n mean sum) c(stat)
    count if inflammatory_words>0 & republicandum==1
restore

********************************************************************************
* TABLE 2 — Baseline change-on-rhetoric (unsigned), full identified controls.
********************************************************************************
reg change_democrats   `UNSIGNED' `PRE' `TOPIC' i.year [aw=sample1], vce(cluster debate_date)
estimates store T2_dem
reg change_republicans `UNSIGNED' `PRE' `TOPIC' i.year [aw=sample1], vce(cluster debate_date)
estimates store T2_rep
* Real result: only discriminatory_words -> Republican change is significant (b~0.184, p~0.013).

********************************************************************************
* TABLE 3 — Source-specific (by speaker party). Central test of H2/H3.
* COMMENT 10: discriminatory_words_d and _r are POSITIVE in the Democratic-change
*             equation (b~0.350*, b~0.223**) — they do NOT mirror inflammatory.
********************************************************************************
reg change_democrats   `SRC' `PRE' `TOPIC' i.year [aw=sample1], vce(cluster debate_date)
estimates store T3_dem
reg change_republicans `SRC' `PRE' `TOPIC' i.year [aw=sample1], vce(cluster debate_date)
estimates store T3_rep
* Real result: inflammatory_words_r -> Republican change ~ +0.130** (p~0.018) survives;
*              inflammatory_words_d -> Democratic change ~ +0.354 (p~0.173), NOT significant.

* --- SPECIFICATION SENSITIVITY (reported as honest robustness, not a headline) ---
* The inflammatory_words_d coefficient is NOT stable across control sets:
reg change_democrats `SRC' `PRE' i.year [aw=sample1], vce(cluster debate_date)          // ~ -1.09***
reg change_democrats `SRC' `PRE' `TOPIC' i.year [aw=sample1], vce(cluster debate_date)   // ~ +0.354 (ns)
* => the sign/significance of the Democratic inflammatory term depends on whether the
*    three topic indicators are included. This fragility is disclosed in the text.

********************************************************************************
* TABLE 4 — Sample-size heterogeneity.
* COMMENTS 7 & 11: inflammatory and discriminatory interactions behave DIFFERENTLY,
* and BOTH discriminatory x High_Sample interactions are negative (not opposing signs).
********************************************************************************
reg change_democrats   c.(`SRC')##i.high_sample1 `PRE' `TOPIC' i.year [aw=sample1], vce(cluster debate_date)
estimates store T4_dem
reg change_republicans c.(`SRC')##i.high_sample1 `PRE' `TOPIC' i.year [aw=sample1], vce(cluster debate_date)
estimates store T4_rep
* Real (DV=Dem change): inflam_r x HS ~ -0.224**; discrim_d x HS ~ -0.289**; discrim_r x HS ~ -0.208***.

********************************************************************************
* TABLE 5 — Polarization amplification (H4).
* COMMENTS 13 & 14: with only 4–5 cycles, High_Polarization (2016/2020) and the linear
* Polarization_Trend (year-2004) are ABSORBED by i.year; their interactions with
* rhetoric are dropped for collinearity. H4 is therefore NOT identified in this design.
********************************************************************************
gen highpol = inlist(year,2016,2020)
reg change_democrats   c.(`SRC')##i.highpol `PRE' `TOPIC' i.year [aw=sample1], vce(cluster debate_date)
* Stata will drop the rhetoric#highpol interactions as collinear with i.year. Confirm:
reg change_democrats   c.(`SRC')##i.highpol `PRE' `TOPIC' i.year [aw=sample1], vce(cluster debate_date) noomitted
* (If you instead drop i.year to "identify" these terms, the estimates merely reload the
*  cross-cycle differences onto the polarization dummy and are not separable from cohort
*  effects — see manuscript Section 6.5. We therefore do not report H4 as a finding.)

********************************************************************************
* COMMENT 8 — Debate-level (~candidate-debate) aggregation, presented in-text.
********************************************************************************
preserve
    collapse (mean) change_democrats change_republicans ///
             (first) `UNSIGNED' inflammatory_words_d inflammatory_words_r year, by(debate_date candidate)
    count
    reg change_republicans `UNSIGNED', vce(robust)   // inflammatory_words ~ +0.132** (p~0.049)
    reg change_democrats   `UNSIGNED', vce(robust)
restore
* Real result: at the debate level only Republican-side inflammatory rhetoric remains
* significant; the other coefficients lose significance, consistent with effective N≈debates.

********************************************************************************
* COMMENT 9 — Formal placebo-difference test.
* Stack the post-debate change and the pre-debate (placebo) change; interact rhetoric
* with POST. The interaction = (post coefficient - placebo coefficient).
********************************************************************************
preserve
    keep debate_date sample1 `PRE' `TOPIC' inflammatory_words_r discriminatory_words_r change_republicans d_republicans
    gen long _row = _n
    expand 2
    bysort _row: gen POST = (_n==1)
    gen y = cond(POST==1, change_republicans, d_republicans)
    drop if missing(y)
    reg y c.inflammatory_words_r##i.POST c.discriminatory_words_r##i.POST `PRE' `TOPIC' [aw=sample1], vce(cluster debate_date)
restore
* Real result: inflammatory_words_r x POST ~ +0.145 (p~0.20, NOT significant);
*              discriminatory_words_r x POST ~ -0.077* (p~0.075).
* => "placebo ~ 0" is NOT the same as "post differs from placebo"; the inflammatory
*    difference is not statistically distinguishable. Stated honestly in the text.

********************************************************************************
* COMMENT 4 — Measurement-error / classification-uncertainty sensitivity.
* Random-label-noise simulation (Stata version of the manuscript's Monte Carlo).
********************************************************************************
reg change_republicans `SRC' `PRE' `TOPIC' i.year [aw=sample1], vce(cluster debate_date)
scalar b0 = _b[inflammatory_words_r]
di "baseline inflammatory_words_r = " b0
* Monte Carlo: perturb counts by +/-1 with prob 0.15, re-estimate, summarize sign stability.
set seed 7
tempname M
postfile `M' double bhat using mc_results.dta, replace
forvalues k = 1/400 {
    preserve
        gen double _noise = (runiform()<0.15)*(round(runiform())*2-1)
        replace inflammatory_words_r = max(0, inflammatory_words_r + _noise)
        quietly reg change_republicans `SRC' `PRE' `TOPIC' i.year [aw=sample1], vce(cluster debate_date)
        post `M' (_b[inflammatory_words_r])
    restore
}
postclose `M'
preserve
    use mc_results.dta, clear
    summarize bhat, detail
    count if sign(bhat)==sign(b0)
    di "share same sign as baseline = " r(N)/_N
restore
* Real result: robust to RANDOM noise (mean ~ +0.126; ~100% same sign). NOTE: it is NOT
* robust to SYSTEMATIC re-classification — re-coding the transcripts with a transparent
* lexicon flips several signs (see code/reclassify_transcripts.py). This is disclosed.

********************************************************************************
* COMMENT 13 — VIF diagnostic: macro vars vs election-year structure.
********************************************************************************
reg change_democrats `SRC' incumbent_democrat gdpbillionsofus percapitaus `PRE' `TOPIC' [aw=sample1]
estat vif
* Real result: VIF for gdpbillionsofus / percapitaus are large relative to the rhetoric
* terms; each macro var takes only one value per year. Confirms they cannot be separated
* from election-year FE.

********************************************************************************
* End. Numbers above are reproduced by code/reanalysis_master.py (Python, no Stata
* license required) and saved to Revised_R2/REAL_RESULTS_R2.txt.
********************************************************************************
