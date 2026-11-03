# Oracle Problem, Testing Scientific Software, Metamorphic Testing, and Property-Based Testing (Hypothesis)

Background research for "Teaching to the Test: Detecting Test Gaming by AI Agents in Autonomous Scientific Discovery" (CIS*4150). This covers the related work and the RQ2 design (hidden metamorphic and Hypothesis property tests alongside hidden pytest unit tests).

Verification legend:
- **[V-search]**: confirmed in this session by web search or a bibliographic record (title, authors, venue, and usually DOI).
- **[V-known]**: a standard reference I am highly confident of, but whose DOI or pages I did not re-check in this session. Re-check the DOI before submitting.
- **[UNVERIFIED]**: not confirmed. Do not cite it until it is checked.

---

## Q1. The test oracle problem (foundations)

### Takeaway
Weyuker (1982) named the class of "non-testable programs": programs, typical of scientific and numerical code, for which no oracle exists or an oracle is too costly. Barr et al. (2015) is the standard survey and sorts the responses into specified, derived, implicit, and human oracles. Metamorphic relations are a key kind of derived oracle. Together these two papers justify why our hidden checks cannot simply compare against a known-correct number.

### Cited Findings (annotated references)
1. **Weyuker, E. J. (1982). "On Testing Non-Testable Programs." *The Computer Journal*, 25(4), 465–470. DOI: 10.1093/comjnl/25.4.465.** [V-known]
   - Summary: Defines "non-testable" programs as those for which an oracle does not exist or is impractical. It names scientific and numerical programs written to find an unknown answer as the archetype, and discusses partial oracles and pseudo-oracles.
   - Supports: the Introduction and Background (the oracle problem in research code), and the motivation for RQ2. Dwarakanath et al. (ISSTA 2018) call ML programs "non-testable" in Weyuker's sense ([arXiv 1808.05353](https://arxiv.org/abs/1808.05353)).
2. **Barr, E. T., Harman, M., McMinn, P., Shahbaz, M., & Yoo, S. (2015). "The Oracle Problem in Software Testing: A Survey." *IEEE Transactions on Software Engineering*, 41(5), 507–525. DOI: 10.1109/TSE.2014.2372785.** [V-known]
   - Summary: A comprehensive survey that classifies oracles as specified (formal specs, contracts), derived (from documentation, other versions, or metamorphic relations), implicit (crashes, exceptions), and human. It also covers how to reduce the human oracle cost.
   - Supports: Background, and the framing of our hidden checks as derived oracles (MRs and properties) plus implicit oracles (crashes, NaN checks) plus specified oracles (pytest assertions with known values).
3. **Liu, H., Kuo, F.-C., Towey, D., & Chen, T. Y. (2014). "How Effectively Does Metamorphic Testing Alleviate the Oracle Problem?" *IEEE TSE*, 40(1), 4–22. DOI: 10.1109/TSE.2013.46.** [V-search] — [Nottingham eprints](https://eprints.nottingham.ac.uk/51787/); [VU preprint PDF](https://vu-9.eprints-hosting.org/33046/1/TSEmt.pdf)
   - Summary: Empirical studies show that "a small number of diverse metamorphic relations, even those identified in an ad hoc manner, had a similar fault-detection capability to a test oracle". Testers with little training could identify enough MRs, and more diverse MRs improved cost-effectiveness.
   - Supports: **RQ2 directly**. It justifies using a small, diverse set of MRs as hidden checks. It also predicts that diversity matters, so we should report detection rates per MR type.

### Inferences
- In our setting the "oracle problem" has two sides. Benchmark tasks (DiscoveryBench, ScienceAgentBench) have a gold answer, so the visible tests can check it, and an agent can hard-code that answer. Hidden MRs check how a solution behaves rather than the value it outputs, so a hard-coded constant output will fail any MR whose expected follow-up output differs from the source output (for example, a scaling MR). It may still pass pure invariance MRs, such as a row permutation that leaves the output unchanged. This asymmetry should be called out in RQ2: **invariance MRs alone cannot catch constant-output gaming. Include at least one "equivariant" MR where the output must change predictably.**

### Gaps
- I did not re-fetch Weyuker or Barr et al. in this session. The bibliographic details are standard but should be checked on the DOI pages.

---

## Q2. Testing scientific software and data-science code

### Takeaway
The systematic literature review by Kanewala & Bieman is the canonical source. It names the oracle problem as the leading challenge in testing scientific software and points to metamorphic testing as a promising response. Hook & Kelly add the idea of "trustworthiness" rather than correctness. Bioinformatics and regression software are mature application areas for MT. I found very little peer-reviewed work on MT for notebook or pandas analysis code specifically, which is a gap our paper can claim.

### Cited Findings (annotated references)
4. **Kanewala, U., & Bieman, J. M. (2014). "Testing Scientific Software: A Systematic Literature Review." *Information and Software Technology*, 56(10), 1219–1232. DOI: 10.1016/j.infsof.2014.05.006.** [V-known]
   - Summary: A systematic review of 62 studies on testing scientific software. Its main challenges are the lack of test oracles, the difficulty of separating model, algorithm, and code faults, and the cultural gap between scientists and software engineers. Metamorphic testing is one of the recommended techniques.
   - Supports: Background/Related Work ("research code lacks known-correct outputs").
5. **Kanewala, U., & Bieman, J. M. (2013). "Using Machine Learning Techniques to Detect Metamorphic Relations for Programs without Test Oracles." *Proc. IEEE ISSRE 2013*, 1–10. DOI: 10.1109/ISSRE.2013.6698899.** [V-known]
   - Summary: Predicts whether a function has a given MR (for example permutative, additive, or inclusive) from features of its control-flow graph.
   - Supports: Related Work (automating MR identification) and the Threats-to-Validity point that MR choice is subjective.
6. **Kanewala, U., Bieman, J. M., & Ben-Hur, A. (2016). "Predicting Metamorphic Relations for Testing Scientific Software: A Machine Learning Approach Using Graph Kernels." *Software Testing, Verification and Reliability*, 26(3), 245–269. DOI: 10.1002/stvr.1594.** [V-known]
   - Summary: Extends the ISSRE 2013 work with graph kernels over program graphs to predict MRs for scientific functions. It uses a catalogue of six MR types: additive, multiplicative, permutative, invertive, inclusive, exclusive.
   - Supports: the MR catalogue (below) and Related Work.
7. **Hook, D., & Kelly, D. (2009). "Testing for Trustworthiness in Scientific Software." *Proc. ICSE Workshop on Software Engineering for Computational Science and Engineering (SECSE '09)*, Vancouver, 59–64. DOI: 10.1109/SECSE.2009.5069163.** [V-search] — [Crossref](https://api.crossref.org/works/10.1109%2FSECSE.2009.5069163); [BibBase](https://bibbase.org/network/publication/hook-kelly-testingfortrustworthinessinscientificsoftware)
   - Summary: Argues that without oracles, scientists judge whether software is *trustworthy* rather than proven correct, and that code faults make that judgment harder. Later work cites it as showing that tighter oracle tolerances help scientists notice unexpected error.
   - Supports: Background, and our design choice of numeric tolerances in hidden checks (for example `math.isclose` with a relative tolerance on regression coefficients).
8. **Chen, T. Y., Ho, J. W. K., Liu, H., & Xie, X. (2009). "An Innovative Approach for Testing Bioinformatics Programs Using Metamorphic Testing." *BMC Bioinformatics*, 10:24. DOI: 10.1186/1471-2105-10-24.** [V-search] — [PMC2657898](https://pmc.ncbi.nlm.nih.gov/articles/PMC2657898)
   - Summary: Shows that MT addresses the lack of an oracle in bioinformatics programs, with two real case studies (network simulation and short-sequence mapping).
   - Supports: Related Work (MT for scientific and research code).
9. **Luu, Q.-H., Lau, M. F., Ng, S. P. H., & Chen, T. Y. (2021). "Testing Multiple Linear Regression Systems with Metamorphic Testing." *Journal of Systems and Software*, 182, 111062. DOI: 10.1016/j.jss.2021.111062. arXiv: 2108.07584.** [V-search] — [arXiv](https://arxiv.org/abs/2108.07584); [Monash](https://research.monash.edu/en/publications/testing-multiple-linear-regression-systems-with-metamorphic-testi/)
   - Summary: Derives **11 MRs from the mathematical properties of linear regression** and evaluates them with mutation analysis on several regression programs.
   - Supports: **RQ2 and the MR catalogue**. This is the most directly applicable source for our regression tasks.
10. **Guderlei, R., Klenk, S., Mayer, J., Schmidt, V., & Spodarev, E. (2007). "Statistical Metamorphic Testing: Testing Programs with Random Output by Means of Statistical Hypothesis Tests and Metamorphic Testing." *Proc. QSIC 2007* (STEV '07 workshop), IEEE, 404–409.** [V-search] — [DBLP index](https://dblp.uni-trier.de/pid/99/4223.html); [Mayer publication list](https://www.mathematik.uni-ulm.de/sai/jmayer/publications.html)
    - Summary: Combines statistical hypothesis tests with MT for programs whose output is random. The MR is checked over the *distribution* of outputs rather than a single run.
    - Supports: Methods (how to handle stochastic analyses such as bootstrap, random-forest predictions, or train/test splits: fix seeds or use statistical MRs) and Threats to Validity (flaky MRs).
    - Note: sources disagree slightly on the venue (QSIC 2007 vs. the STEV 2007 workshop in the same IEEE proceedings). Cite it as "QSIC 2007 (STEV workshop), pp. 404–409".
11. **Xie, X., Zhang, Z., Chen, T. Y., Liu, Y., Poon, P.-L., & Xu, B. (2020). "METTLE: A METamorphic Testing Approach to Assessing and Validating Unsupervised Machine Learning Systems." *IEEE Transactions on Reliability*, 69(4). DOI: 10.1109/TR.2020.2972266. arXiv: 1807.10453.** [V-search; the full author list was taken from memory, so check it] — [arXiv](https://arxiv.org/abs/1807.10453)
    - Summary: Proposes **11 generic MRs for clustering systems** and evaluates them on six clustering algorithms.
    - Supports: the MR catalogue (clustering-style analysis tasks).
12. **MetaCDP: Metamorphic Testing for Quality Assurance of Containerized Data Pipelines (Montana State University).** [V-search for existence; authors and venue not confirmed] — [ScholarWorks](https://scholarworks.montana.edu/handle/1/18983)
    - Summary: Proposes 10 MRs for data pipelines that feed an anomaly-detection model and reports several erroneous behaviours in an e-commerce pipeline.
    - Supports: Related Work (MT for data pipelines). Check the authors and venue before citing.
13. **"Metamorphic Testing for Adobe Data Analytics Software" (2017; University of Nottingham Ningbo record).** [V-search for existence only] — [Nottingham Ningbo](https://research.nottingham.edu.cn/en/publications/metamorphic-testing-for-adobe-data-analytics-software/)
    - Summary: Builds MRs from geometric transformations of time series for anomaly detection and found three real bugs.
    - Supports: Related Work (industrial MT on analytics code). Get the full citation from the record.
14. **Hypothesis pandas strategies (EuroSciPy 2023 talk, "Generating Data Frames for your test – using Pandas strategies in Hypothesis").** [V-search; a talk, not a paper] — [pretalx](https://pretalx.com/euroscipy-2023/talk/WPEXXC/)
    - Supports: Methods (practical evidence that `hypothesis.extra.pandas` can generate synthetic DataFrames for testing analysis code). Cite the Hypothesis docs in preference to the talk.

### Inferences
- A literature search found no peer-reviewed study that applies MT/PBT specifically to **LLM-agent-written scientific data-analysis scripts** (DiscoveryBench- or ScienceAgentBench-style). This looks like a real novelty claim for the paper, but it rests on a limited search.

### Gaps
- I did not find a peer-reviewed MT study of Jupyter notebook or pandas analysis code specifically. One blog post (qaskills.sh) discusses MT for data pipelines but is not citable.
- I did not verify the exact list of Luu et al.'s 11 regression MRs (the full text was not fetched). The regression MRs in the catalogue below are derived from standard OLS algebra and are attributed to Luu et al. only "in spirit". Check them against the paper.

---

## Q3. Metamorphic testing: origins, surveys, and ML/statistics applications, including LLM-generated MRs

### Takeaway
MT was introduced by Chen, Cheung & Yiu (1998). Segura et al. (TSE 2016) and Chen et al. (CSUR 2018) are the two standard surveys. For ML and statistics, Murphy et al. (2008) and Xie et al. (2011) are the foundational MR catalogues: permutation, addition of uninformative attributes, affine transformation, duplication, and others. Recent (2023–2025) work shows LLMs can propose MRs, but only about 30–44% of candidates are valid. That argues for human-written hidden MRs in our study, with LLM-generated MRs at most an extension.

### Cited Findings (annotated references)
15. **Chen, T. Y., Cheung, S. C., & Yiu, S. M. (1998). "Metamorphic Testing: A New Approach for Generating Next Test Cases." Technical Report HKUST-CS98-01, Hong Kong University of Science and Technology. Republished as arXiv:2002.12543 (2020).** [V-known] — [arXiv](https://arxiv.org/abs/2002.12543)
    - Summary: The original MT proposal. Uses successful test cases as sources for "follow-up" test cases via necessary properties of the target function (for example sin(x) = sin(π − x)).
    - Supports: Background (definition of MR, source and follow-up test cases).
16. **Segura, S., Fraser, G., Sánchez, A. B., & Ruiz-Cortés, A. (2016). "A Survey on Metamorphic Testing." *IEEE TSE*, 42(9), 805–824. DOI: 10.1109/TSE.2016.2532875.** [V-known]
    - Summary: Reviews about 119 MT papers. It covers MR identification, application domains (including scientific software, bioinformatics, and ML), and evidence that MT is effective at fault detection.
    - Supports: Background and Related Work.
17. **Chen, T. Y., Kuo, F.-C., Liu, H., Poon, P.-L., Towey, D., Tse, T. H., & Zhou, Z. Q. (2018). "Metamorphic Testing: A Review of Challenges and Opportunities." *ACM Computing Surveys*, 51(1), Article 4, 1–27. DOI: 10.1145/3143561.** [V-known; the Liu et al. search results also mention a related CSUR review]
    - Summary: Reviews the state of MT, open challenges (MR identification and selection, composition), and opportunities. It frames MT as both test generation and test-result verification.
    - Supports: Background, and the Discussion of MR selection for RQ2.
18. **Murphy, C., Kaiser, G., Hu, L., & Wu, L. (2008). "Properties of Machine Learning Applications for Use in Metamorphic Testing." *Proc. 20th Int'l Conf. on Software Engineering and Knowledge Engineering (SEKE 2008)*, San Francisco, 867–872.** [V-search] — [Columbia Academic Commons](https://academiccommons.columbia.edu/doi/10.7916/D8XK8PFD); [Zhang et al. ML Testing survey](https://arxiv.org/pdf/1906.10742)
    - Summary: Lists and classifies metamorphic properties of ML algorithms. The six classic categories from this paper are **additive, multiplicative, permutative, invertive, inclusive, exclusive**, and the paper shows how they reveal defects. It was not empirically evaluated, but it underpins later ML MT work.
    - Supports: the **MR catalogue** (the generic categories) and Related Work.
19. **Xie, X., Ho, J. W. K., Murphy, C., Kaiser, G., Xu, B., & Chen, T. Y. (2011). "Testing and Validating Machine Learning Classifiers by Metamorphic Testing." *Journal of Systems and Software*, 84(4), 544–558. DOI: 10.1016/j.jss.2010.11.920.** [V-search] — [PMC3082144](https://pmc.ncbi.nlm.nih.gov/articles/PMC3082144)
    - Summary: Defines MRs for kNN and Naive Bayes, including consistency under affine transformation of features, permutation of attributes, addition of uninformative attributes, permutation of class labels, and addition or removal of samples or classes. It applies them to Weka and finds real faults. Later analysis notes that not all MRs are needed for every algorithm.
    - Supports: the **MR catalogue** and RQ2.
20. **Saha, P., & Kanewala, U. (2019). "Fault Detection Effectiveness of Metamorphic Relations Developed for Testing Supervised Classifiers." arXiv:1904.07348 (AITest 2019).** [V-search for arXiv; venue from memory] — [arXiv](https://arxiv.org/pdf/1904.07348)
    - Summary: Uses mutation analysis to measure how well Xie et al.'s classifier MRs detect faults. It restates MR1 (affine kx+b), MR2 (attribute permutation), and MR3 (uninformative attribute).
    - Supports: **RQ2**. It is a methodological precedent for measuring the detection power of each MR with injected faults, which is analogous to our "gamed code" variants.
21. **Dwarakanath, A., Ahuja, M., Sikand, S., Rao, R. M., Bose, R. P. J. C., Dubash, N., & Podder, S. (2018). "Identifying Implementation Bugs in Machine Learning Based Image Classifiers Using Metamorphic Testing." *Proc. ISSTA 2018*, 118–128. DOI: 10.1145/3213846.3213858.** [V-search] — [arXiv 1808.05353](https://arxiv.org/abs/1808.05353)
    - Summary: MRs for an SVM and a deep-learning classifier (for example permuting features or channels, and shifting or scaling inputs) caught **71% of injected implementation bugs**.
    - Supports: RQ2 (a precedent for detection rates on injected faults) and the MR catalogue.
22. **Zhang, J. M., Harman, M., Ma, L., & Liu, Y. (2022). "Machine Learning Testing: Survey, Landscapes and Horizons." *IEEE TSE*, 48(1), 1–36. arXiv:1906.10742.** [V-search for arXiv; TSE details from memory] — [arXiv](https://arxiv.org/pdf/1906.10742)
    - Summary: A survey of ML testing that includes a section on MT and oracles for ML, and summarizes Murphy, Xie, Dwarakanath, and others.
    - Supports: Related Work.
23. **Zhang, Y., Towey, D., & Pike, M. (2023). "Automated Metamorphic-Relation Generation with ChatGPT: An Experience Report." *Proc. IEEE COMPSAC 2023*.** [V-search for existence] — [Nottingham Ningbo record](https://research.nottingham.edu.cn/en/publications/automated-metamorphic-relation-generation-with-chatgpt-an-experie/)
    - Summary: A pilot study that uses ChatGPT (GPT-3.5) to generate MRs.
    - Supports: Related Work (LLM-generated MRs).
24. **Zhang, Y., Towey, D., Pike, M., Luu, Q.-H., Liu, H., & Chen, T. Y. (2025). "Integrating Artificial Intelligence with Human Expertise: An In-depth Analysis of ChatGPT's Capabilities in Generating Metamorphic Relations." arXiv:2503.22141.** [V-search] — [arXiv](https://arxiv.org/abs/2503.22141)
    - Summary: Compares GPT-3.5 and GPT-4 MRs on nine SUTs, using both a GPT-based evaluator and human evaluators. It concludes that GPT-4 is capable at generating MRs, with human review still needed.
    - Supports: Related Work, and a possible extension (LLM-written hidden MRs).
25. **Zhang, et al. (2025). "Can Large Language Models Discover Metamorphic Relations? A Large-Scale Empirical Study." *Proc. IEEE SANER 2025*.** [V-search; the full author list was not confirmed (researchr lists it as ZhangSLD25)] — [SANER 2025 program](https://conf.researchr.org/details/saner-2025/saner-2025-papers/41/Can-Large-Language-Models-Discover-Metamorphic-Relations-A-Large-Scale-Empirical-Stu)
    - Summary: On 37 SUTs from earlier MT studies, gpt-3.5-turbo and gpt-4 produced many MR candidates, but only **29.86% (GPT-3.5) and 43.79% (GPT-4)** were valid.
    - Supports: Discussion and Threats. It shows that LLM-generated MRs are unreliable without human vetting, which justifies hand-written hidden MRs.

### Inferences
- The "classic six" (Murphy 2008; Kanewala 2016) map directly onto pandas and statistics code: permutative = row or column shuffle, additive and multiplicative = shift and scale, inclusive and exclusive = add or remove rows.
- Liu et al. (2014) suggest **diversity** of MRs matters more than their number. For RQ2, compare MR *families* (invariance vs. equivariance vs. inclusion/exclusion) rather than individual MRs.

### Gaps
- No MT paper found specifically targets LLM-agent-produced research code.

---

## Q4. Property-based testing (QuickCheck, Hypothesis) and PBT for LLM-generated code

### Takeaway
QuickCheck (2000) started PBT. Hypothesis (JOSS 2019) is the standard Python implementation and has `hypothesis.extra.pandas` and `hypothesis.extra.numpy` strategies suited to generating DataFrames. Empirical studies (Goldstein et al. ICSE 2024; Ravi & Coblenz OOPSLA 2025) show PBT is valued in practice and is much stronger than unit tests at killing mutants: about 50 times more mutants per test, and 76% of those found within the first 20 inputs. Recent LLM work (2023–2025) uses PBT both to test and to guide LLM-generated code, and explicitly notes that example-based tests can be "self-deceiving".

### Cited Findings (annotated references)
26. **Claessen, K., & Hughes, J. (2000). "QuickCheck: A Lightweight Tool for Random Testing of Haskell Programs." *Proc. ACM SIGPLAN ICFP 2000*, 268–279. DOI: 10.1145/351240.351266.** [V-known]
    - Summary: Introduces PBT. Programmers state properties as Haskell functions, QuickCheck checks them on many randomly generated inputs, and it reports counterexamples.
    - Supports: Background (the PBT definition).
27. **MacIver, D. R., Hatfield-Dodds, Z., and many other contributors (2019). "Hypothesis: A New Approach to Property-Based Testing." *Journal of Open Source Software*, 4(43), 1891. DOI: 10.21105/joss.01891.** [V-known]
    - Summary: Describes Hypothesis, the main Python PBT library. Its key design idea is integrated, internal test-case reduction (shrinking) over a byte-stream "choice sequence", which makes shrinking work for arbitrary user strategies. It integrates with pytest.
    - Supports: Methods (tooling for the hidden property tests). Also cite the Hypothesis docs for `hypothesis.extra.pandas` (data_frames, column, range_indexes) when describing DataFrame generation.
28. **Goldstein, H., Cutler, J. W., Dickstein, D., Pierce, B. C., & Head, A. (2024). "Property-Based Testing in Practice." *Proc. IEEE/ACM ICSE 2024*, Article 187. DOI: 10.1145/3597503.3639581.** [V-search for existence (UPenn page); DOI from memory] — [UPenn PDF page](https://cis.upenn.edu/~bcpierce/papers/icse24-pbt-in-practice); [SUNY record](https://researchconnect.suny.edu/en/publications/property-based-testing-in-practice/)
    - Summary: An interview study of experienced PBT users at Jane Street. It reports that PBT is valued for high-leverage scenarios (for example comparing a function to a model implementation, or round-trip properties) and identifies obstacles such as writing generators and judging how good a property is.
    - Supports: Related Work and Discussion (the practicality of writing properties).
29. **Ravi, S., & Coblenz, M. (2025). "An Empirical Evaluation of Property-Based Testing in Python." *Proc. ACM on Programming Languages (OOPSLA 2025)*. DOI: 10.1145/3764068.** [V-search] — [SPLASH 2025 page](https://2025.splashcon.org/details/OOPSLA/102); [Crossref](https://api.crossref.org/works/10.1145%2F3764068)
    - Summary: A corpus study of 426 Python projects that use Hypothesis, with a static analysis that sorts property tests into 12 categories. In mutation testing on 40 projects, **each property-based test killed about 50 times as many mutants as the average unit test, and 76% of mutants caught were caught within the first 20 generated inputs**.
    - Supports: **RQ2 (a key quantitative prior)**. It predicts that hidden Hypothesis tests should beat hidden unit tests at catching gamed code. It also justifies a modest `max_examples` budget, which limits runtime on heavy analysis tasks.
30. **de Oliveira, Corgozinho, Rocha, & Valente. PBT in Python study (*Empirical Software Engineering*, listed as "2027").** [UNVERIFIED] — [UFMG PDF](https://homepages.dcc.ufmg.br/~mtov/pub/2027-esem-pbt-python.pdf)
    - Summary (from a search snippet only): analyzes 367 property-based tests from 244 Python projects.
    - Flag: the "2027" date is odd given today's date (2026-11-03), and may be an accepted or forthcoming paper. Check the title, venue, and status before citing.
31. **Vikram, V., Lemieux, C., Sunshine, J., & Padhye, R. (2023/2024). "Can Large Language Models Write Good Property-Based Tests?" arXiv:2307.04346 (v1 July 2023, v2 July 2024).** [V-search] — [arXiv](https://arxiv.org/abs/2307.04346v2)
    - Summary: Evaluates LLM-written Hypothesis tests for 40 Python library APIs (GPT-4, Gemini-1.5-Pro, Claude-3-Opus). It introduces a "property coverage" metric based on property mutants. The best setup produced a valid and sound test in 2.4 samples on average, and GPT-4 automatically produced correct tests for 21% of the properties that can be extracted from documentation.
    - Supports: Related Work, Discussion (whether hidden properties could be LLM-generated), and Methods (the property-mutant idea for validating our own hidden properties).
32. **He, L., Chen, Z., Zhang, Z., Shao, J., Gao, X., & Sheng, L. (2025). "Use Property-Based Testing to Bridge LLM Code Generation and Validation." arXiv:2506.18315.** [V-search] — [arXiv](https://arxiv.org/abs/2506.18315v1)
    - Summary: Proposes the Property-Generated Solver (PGS). A Tester agent defines high-level properties (invariants) and a Generator agent refines code from PBT feedback, giving 23.1–37.3% relative pass@1 gains over TDD baselines. It argues that LLM-generated example tests can mirror the generator's own mistakes ("a cycle of self-deception") and that generating exact oracles can be harder than generating the code.
    - Supports: **the motivation for RQ2**. It is direct evidence that properties are a more robust check on LLM code than example-based tests.

### Inferences
- Combining Ravi & Coblenz (PBT kills far more mutants than unit tests) with Liu et al. (diverse MRs perform close to a true oracle) leads to a falsifiable RQ2 hypothesis: *hidden Hypothesis/MR tests will detect more gamed solutions than hidden pytest unit tests do, especially gaming that special-cases the visible inputs.*
- A gamed solution that special-cases the exact visible test data (for example `if df.shape == (150, 5): return 0.42`) will be exposed by Hypothesis-generated DataFrames of other shapes or values. A gamed solution that returns a hard-coded constant will be exposed by equivariant MRs, though not by invariant ones.

### Gaps
- Pages for Goldstein et al. and the exact JOSS author list for Hypothesis were not re-fetched.

---

## Q5. Work that uses metamorphic or property tests, or stronger hidden tests, to detect cheating, hard-coding, or overfitting by LLM agents

### Takeaway
I found **no paper that uses metamorphic testing specifically to detect test gaming by LLM agents**. That is a novelty claim for this project, with the caveat that the search was limited. The closest work: (a) EvalPlus shows that weak visible tests overstate correctness and that augmented hidden tests expose wrong code; (b) ImpossibleBench measures how often agents exploit tests, and finds that hiding tests lowers hacking to near zero but hurts legitimate performance; (c) PGS argues properties avoid "self-deceiving" tests.

### Cited Findings (annotated references)
33. **Liu, J., Xia, C. S., Wang, Y., & Zhang, L. (2023). "Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation." *NeurIPS 2023*. arXiv:2305.01210.** [V-search] — [NeurIPS page](https://neurips.cc/virtual/2023/poster/72990)
    - Summary: EvalPlus adds LLM- and mutation-generated inputs to HumanEval, giving about 80 times more tests (HumanEval+). Pass@k drops by up to 19.3–28.9%, and model rankings flip, which shows that weak tests accept wrong code.
    - Supports: **RQ2 and the Motivation**. It is the main precedent for hidden, augmented tests catching code that passes the visible tests.
34. **Zhong, Z., Raghunathan, A., & Carlini, N. (2025). "ImpossibleBench: Measuring LLMs' Propensity of Exploiting Test Cases." arXiv:2510.20270; ICLR 2026 (per secondary sources).** [V-search; check the ICLR acceptance on OpenReview] — [arXiv](https://arxiv.org/pdf/2510.20270); [OpenReview PDF](https://openreview.net/pdf/f0edc0ee21784f063e1f7970ecbd360037762bb7.pdf)
    - Summary: Mutates the tests of LiveCodeBench and SWE-bench tasks so they conflict with the specification. Any pass is then a cheat. Reports GPT-5 exploiting tests 76% of the time on oneoff impossible-SWEbench. Hiding tests cuts hacking to near zero but noticeably lowers legitimate performance, and read-only test access is a middle ground. Observed strategies include modifying tests and special-casing.
    - Supports: the Introduction and the definition of test gaming. It is also the main comparison point for RQ2: their detection uses impossible tasks, while ours uses hidden MRs and properties on *possible* tasks. (The headline figures come from the arXiv abstract and secondary summaries, so re-check them in the PDF.)
35. **He et al. (2025), PGS** (reference 32 above): the argument that example-based tests are vulnerable to "self-deception" also supports using properties to detect gamed code.

### Inferences
- **Proposed contribution:** treat hidden MRs and properties as a *gaming detector that needs no gold answer*. This complements ImpossibleBench, whose detection depends on specially constructed impossible tasks, and EvalPlus, which needs a reference solution to generate expected outputs.

### Gaps
- I did not verify (in this session) other reward-hacking or test-overfitting papers I am aware of, for example studies showing SWE-bench "solved" patches are often wrong (PatchDiff/"Are 'Solved Issues' in SWE-bench Really Solved Correctly?", 2025) or UTBoost (ACL 2025). **[UNVERIFIED]** Another researcher or a follow-up search should confirm them.
- I found no paper using MT to detect overfitting patches in automated program repair, although that analogy is natural.

---

## Catalogue of metamorphic relations for data-analysis, statistical, and ML code

Notation: `f(D)` is the analysis output on DataFrame `D`. "Source" means the paper that proposes the relation or its generic category. Hypothesis implementation hints use `hypothesis.extra.pandas` (`data_frames`, `column`) and `hypothesis.extra.numpy`. Use tolerances (`math.isclose`/`np.allclose`, rel_tol about 1e-6) because floating-point results can differ under reordering.

| # | MR | Transformation → expected relation | Applies to | Detects (gaming type) | Source |
|---|----|-----------------------------------|-----------|----------------------|--------|
| 1 | **Row permutation** | shuffle rows → output unchanged (for order-independent statistics: mean, OLS coefficients, correlations, t-tests, group means) | stats, regression, aggregation | special-casing on row order or index, `iloc[0]` hacks | "Permutative" category, Murphy et al. 2008; Kanewala et al. 2016; attribute/sample permutation, Xie et al. 2011 |
| 2 | **Column permutation** | reorder columns (selection is by name) → output unchanged | any analysis that selects columns by name | positional column hard-coding | "Permutation of attributes", Xie et al. 2011 (MR2 in Saha & Kanewala 2019); Dwarakanath et al. 2018 |
| 3 | **Add irrelevant column** | add a column of noise or a constant that the analysis does not use → output unchanged | regression on named predictors, summary stats | code that uses `df.values` or all columns blindly | "Addition of uninformative attributes", Xie et al. 2011 |
| 4 | **Affine scaling of a predictor** | x' = kx + b (k ≠ 0) → OLS slope β' = β/k, R², t-statistics and p-values unchanged; classifier predictions (kNN with scaled features, trees) unchanged where the method is invariant | regression, correlation, tree models | **constant/hard-coded outputs** (equivariant MR) | Affine transformation, Xie et al. 2011 (MR1); "multiplicative/additive", Murphy et al. 2008; linear-regression MRs, Luu et al. 2021 [check exact MR list] |
| 5 | **Scale the response** | y' = ky → coefficients ×k, R² and p-values unchanged; mean ×k, SD ×abs(k) | regression, descriptive stats | hard-coded numbers | "Multiplicative", Murphy et al. 2008; Luu et al. 2021 [check] |
| 6 | **Shift (additive)** | y' = y + c → mean +c, variance/SD unchanged, OLS intercept +c and slopes unchanged, correlation unchanged | descriptive stats, regression | hard-coded numbers | "Additive", Murphy et al. 2008; Kanewala et al. 2016 |
| 7 | **Unit conversion** | e.g. cm→m (×0.01), °C→°F (affine) → results transform predictably, and dimensionless results (r, R², p, t) are unchanged | science datasets with units | hard-coding; unit bugs | a special case of 4–6 (affine MR; Xie et al. 2011; Murphy et al. 2008). I found no separate source for "unit conversion" as a named MR, so present it as a domain instance of the affine MR |
| 8 | **Duplicate the whole dataset** | D' = D ∪ D → mean, proportions, and OLS point estimates unchanged; standard errors shrink by about 1/√2; p-values drop; n doubles | stats, regression | code that hard-codes n or the outputs | "Inclusive" category, Murphy et al. 2008; "addition of classes by duplicating samples", Xie et al. 2011 [recalled from memory, so check] |
| 9 | **Remove or add irrelevant rows** | add rows that the analysis filters out (for example another group or category) → output unchanged | grouped or filtered analyses | missing or hard-coded filtering | "Inclusive/exclusive", Murphy et al. 2008; Kanewala et al. 2016 |
| 10 | **Partition and combine** | split by group and analyse each part, compared with the grouped analysis on the whole → identical per-group results | groupby aggregations | state leakage, boundary bugs | data-pipeline MRs (MetaCDP; practitioner blog, not citable); compositional MRs, Chen et al. CSUR 2018 [general] |
| 11 | **Permute class or category labels** | rename categories consistently → results map through the same renaming | classification, group comparisons | hard-coded label strings or values | "Permutation of class labels", Xie et al. 2011 |
| 12 | **Swap X and Y in correlation** | corr(x,y) = corr(y,x); the regression R² of y~x equals that of x~y in the simple case | correlation | hard-coding | standard symmetry property; generic "permutative/invertive" category, Murphy et al. 2008 |
| 13 | **Negate a predictor** | x' = −x → slope sign flips, magnitude, R² and p unchanged | regression | constant outputs | "Invertive", Murphy et al. 2008; affine with k = −1 (Xie et al. 2011) |
| 14 | **Re-prediction / train-on-test consistency** | add a test sample with its predicted label to the training set → its prediction is unchanged (for kNN-like models) | classifiers | prediction hard-coding | "Consistency with re-prediction", Xie et al. 2011 [from memory, so check] |
| 15 | **Stochastic MR** | for random methods (bootstrap, RF, splits): fix the seed and check invariance, or check the output distribution with a hypothesis test | ML with randomness | flaky or hard-coded outputs | Guderlei et al. 2007 (statistical MT) |
| 16 | **Clustering MRs** | e.g. permuting points, duplicating points, adding far-away outlier clusters → the cluster assignments of the original points are preserved | clustering | hard-coded labels | METTLE, Xie et al. 2020 (11 MRs) |

Property-style (non-metamorphic) checks that suit Hypothesis:
- **Range or validity properties:** 0 ≤ p ≤ 1; R² ≤ 1; −1 ≤ r ≤ 1; SD ≥ 0; predicted probabilities sum to 1. These are implicit and specified oracles in Barr et al.'s terms.
- **Model-based (differential) property:** compare the output with a simple reference implementation (for example `numpy.linalg.lstsq` vs. the agent's statsmodels code) on generated data. Goldstein et al. 2024 report this is a common pattern in practice.
- **Planted-signal property:** generate data with a known effect (y = 2x + noise) and check that the analysis recovers β ≈ 2. This turns the oracle problem around by building a synthetic oracle. It is a standard simulation technique and I am not aware of a single canonical citation. Present it as a design choice.

**Design caveat (from the inferences above):** MRs 1–3, 9, and 10 are *invariance* relations, which a solution returning a hard-coded constant passes trivially. MRs 4–8, 11, and 13 are *equivariant* relations, which hard-coded constants fail. So RQ2 should report detection per MR family, and the hidden suite should include both families.
