# Mutation Testing, Test Adequacy, LLM Test Generation, and Benchmark Contamination

Research date: 2026-11-03. Context: CIS*4150 paper "Teaching to the Test: Detecting Test Gaming by AI Agents in Autonomous Scientific Discovery".

Verification legend:
- **[V]**: confirmed this session from a primary or bibliographic source (arXiv, ACL Anthology, author or publisher page).
- **[K]**: a standard, well-known citation given from prior knowledge. I did not re-check it this session. Confirm the DOI before final submission.

RQ mapping used below. RQ-adequacy means "how strong are our hidden suites (mutation score)". RQ-detection means "which suites detect gamed code". The section tags are Background, Related Work, Methodology and Threats to Validity.

---

## Q1. Mutation testing foundations (DeMillo 1978, Jia & Harman 2011, Papadakis 2019, Just 2014, coupling effect, equivalent mutants)

### Takeaway
Mutation testing rests on two hypotheses: the competent programmer hypothesis and the coupling effect. Both come from DeMillo, Lipton & Sayward (1978), and Offutt tested the coupling effect empirically. Just et al. (2014) gave evidence that mutant detection correlates with real-fault detection independently of coverage, which justifies using mutation score as a proxy for suite strength. The equivalent-mutant problem is the main validity threat to raw mutation scores.

### Cited Findings (annotated references)
1. **DeMillo, R. A., Lipton, R. J., & Sayward, F. G. (1978). "Hints on Test Data Selection: Help for the Practicing Programmer." *IEEE Computer*, 11(4), 34–41. DOI: 10.1109/C-M.1978.218136.** [K]
   - Summary: This is the founding paper of mutation analysis. It introduces the competent programmer hypothesis and the coupling effect: tests that detect simple faults also detect complex ones.
   - Supports: Background (mutation testing origins). It also gives the theoretical basis for treating gamed code as a "mutant" of a correct solution.
2. **Jia, Y., & Harman, M. (2011). "An Analysis and Survey of the Development of Mutation Testing." *IEEE TSE*, 37(5), 649–678. DOI: 10.1109/TSE.2010.62.** [K]. Its existence is corroborated indirectly: Madeyski et al. (2014) cite it as the prior broad mutation-testing review ([Madeyski preprint](https://madeyski.e-informatyka.pl/download/Madeyski14TSE.pdf)).
   - Summary: This is the canonical survey of mutation testing theory, cost reduction, the equivalent-mutant problem, tools and applications up to about 2010.
   - Supports: Background and Related Work.
3. **Papadakis, M., Kintis, M., Zhang, J., Jia, Y., Le Traon, Y., & Harman, M. (2019). "Mutation Testing Advances: An Analysis and Survey." *Advances in Computers*, vol. 112, pp. 275–378. DOI: 10.1016/bs.adcom.2018.03.015.** [V] Sources: [UCL Discovery](https://discovery.ucl.ac.uk/id/eprint/10056704/) and [author PDF](https://mutationtesting.uni.lu/survey.pdf).
   - Year: the publisher's repository gives 2018 and UCL gives 2019. Cite it as 2019, volume 112.
   - Summary: Surveys a decade of advances on the fundamental problems of mutation testing. Covers mutation used to evaluate test-suite adequacy, to guide test generation and to support experiments. Gives best-practice advice for using mutation in empirical studies.
   - Supports: Methodology. Use it to justify the mutation-score procedure, mutant selection and reporting practices.
4. **Just, R., Jalali, D., Inozemtseva, L., Ernst, M. D., Holmes, R., & Fraser, G. (2014). "Are Mutants a Valid Substitute for Real Faults in Software Testing?" *Proc. FSE 2014*, pp. 654–665. DOI: 10.1145/2635868.2635929.** [V] for the citation and findings: [author abstract](https://homes.cs.washington.edu/~mernst/pubs/mutation-effectiveness-fse2014-abstract.html) and [PDF](https://people.cs.umass.edu/~rjust/publ/mutants_real_faults_fse_2014.pdf). [K] for the DOI string.
   - Summary: The study used 357 real faults in 5 Java programs and about 230,000 mutants. Mutant detection correlates significantly with real-fault detection, independently of coverage, and more strongly than statement coverage does. A coupling effect holds for 73% of real faults. About 17% of faults are not coupled to mutants, and about 10% would need new or stronger operators. The paper won an ACM SIGSOFT Distinguished Paper Award.
   - Supports: RQ-adequacy. Justifies mutation score as a suite-strength metric. The roughly 27% uncoupled faults are a Threat to Validity: gamed code may not look like a standard mutant.
5. **Offutt, A. J. (1992). "Investigations of the Software Testing Coupling Effect." *ACM TOSEM*, 1(1), 5–20. DOI: 10.1145/125489.125473.** [V] Sources: [netlib TOSEM bib](https://netlib.org/tex/bib/tosem.html) and [Scilit](https://www.scilit.net/publications/4cbf54516c3a44ccfb7c570f59c4de53).
   - Summary: An empirical study supporting the coupling effect. Test data that kills first-order mutants also kills most higher-order (composite) mutants. It extends Offutt's 1989 TAV paper "The Coupling Effect: Fact or Fiction."
   - Supports: Background and RQ-detection. It is an argument that suites killing simple mutants should also catch more complex gaming.
6. **Madeyski, L., Orzeszyna, W., Torkar, R., & Józala, M. (2014). "Overcoming the Equivalent Mutant Problem: A Systematic Literature Review and a Comparative Experiment of Second Order Mutation." *IEEE TSE*, 40(1), 23–42. DOI: 10.1109/TSE.2013.44.** [V] Source: [author preprint](https://madeyski.e-informatyka.pl/download/Madeyski14TSE.pdf).
   - Summary: The review found 17 techniques in 22 articles, in three categories: detecting, suggesting, and avoiding equivalent mutants. Detection results are "still far from perfect." Second-order mutation, especially the JudyDiffOp strategy, reduced the equivalent-mutant burden.
   - Supports: Threats to Validity. Surviving mutmut mutants may be equivalent, so the mutation score is a lower bound unless survivors are triaged by hand.
   - Note: the undecidability of program equivalence is usually credited to Budd & Angluin (1982), "Two notions of correctness and their relation to testing," *Acta Informatica* 18. [K]. Not re-verified this session.

### Inferences
- Gamed code differs from a correct reference solution, much as a mutant differs from the original program. The paper can treat gaming detection as mutant killing. A hidden suite with a high mutmut score should be more likely to fail gamed solutions. Just et al. and Offutt support this, but because 17–27% of real faults were not coupled to mutants, the claim should be hedged.
- For RQ-adequacy, report the mutation score together with the number of survivors that were inspected for equivalence (Papadakis et al. best practice; Madeyski et al.).

### Gaps
- DOIs for DeMillo 1978 and Jia & Harman 2011 come from memory. Check them on IEEE Xplore.

---

## Q2. Test adequacy (Zhu et al. 1997, Inozemtseva & Holmes 2014, Ammann & Offutt)

### Takeaway
Coverage criteria are a weak proxy for fault-detection ability once suite size is controlled for. Mutation-based adequacy is the stronger criterion. This supports the paper's choice of mutmut over coverage alone.

### Cited Findings (annotated references)
1. **Zhu, H., Hall, P. A. V., & May, J. H. R. (1997). "Software Unit Test Coverage and Adequacy." *ACM Computing Surveys*, 29(4), 366–427. DOI: 10.1145/267580.267590.** [K]
   - Summary: The classic survey of test adequacy criteria. It covers structural, fault-based (including mutation) and error-based criteria, and frames adequacy both as a stopping rule and as a measure of test quality.
   - Supports: Background (definitions of test adequacy).
2. **Inozemtseva, L., & Holmes, R. (2014). "Coverage Is Not Strongly Correlated with Test Suite Effectiveness." *Proc. ICSE 2014*, pp. 435–445. DOI: 10.1145/2568225.2568271.** [V] Sources: [author PDF](https://cs.ubc.ca/~rtholmes/papers/icse_2014_inozemtseva.pdf) and [BibTeX record](https://bibtex.github.io/ICSE-2014-InozemtsevaH.html).
   - Summary: The study used large Java programs and measured effectiveness with mutation testing. Coverage correlated only low to moderately with effectiveness once suite size was controlled for, so coverage should not be a quality target. The paper won an ACM SIGSOFT Distinguished Paper Award, and [UBC](https://www.cs.ubc.ca/award/2024/04/icse-most-influential-paper-award) lists a 2024 ICSE Most Influential Paper award.
   - Supports: Methodology. Explains why the study reports mutation score rather than only line or branch coverage.
3. **Ammann, P., & Offutt, J. (2016). *Introduction to Software Testing* (2nd ed.). Cambridge University Press. DOI: 10.1017/9781316771273.** [K]
   - Summary: The standard textbook. It organises testing around coverage criteria over graphs, logic, input space and syntax, with mutation as syntax-based testing. It also gives the RIPR model (Reachability, Infection, Propagation, Revealability).
   - Supports: Background (terminology), and RIPR can explain why some gamed outputs are not revealed by weak oracles.

### Inferences
- Meta's ACH work found that 49% of its accepted mutation-killing tests would have been discarded under a line-coverage criterion (see Q3). This is independent industrial support for Inozemtseva & Holmes.

### Gaps
- Not checked: the publisher pages for the Zhu et al. DOI and the Ammann & Offutt 2nd-edition DOI. A 3rd edition may exist. Check Cambridge UP.

---

## Q3. Mutation testing with LLMs (2023–2026)

### Takeaway
LLMs are now used both to generate mutants and to generate tests that kill mutants. Meta's ACH is the main industrial example. ACH also shows that LLMs can act as reasonably accurate equivalent-mutant detectors.

### Cited Findings (annotated references)
1. **Foster, C., Gulati, A., Harman, M., et al. (2025). "Mutation-Guided LLM-based Test Generation at Meta." arXiv:2501.12862. Published in *FSE Companion '25*, pp. 180–191.** [V] Sources: [arXiv](https://arxiv.org/abs/2501.12862) and [UCL Discovery](https://discovery.ucl.ac.uk/id/eprint/10218052).
   - Summary: ACH generates mutants from textual descriptions of concerns (here, privacy) and then generates tests that kill them. Applied to 10,795 Android Kotlin classes on 7 platforms, it produced 9,095 mutants and 571 privacy-hardening tests. Engineers accepted 73% of the tests. 49% of the tests would have been discarded under a line-coverage criterion. It uses Llama 3.1 70B. The LLM equivalent-mutant detector reached 0.95 precision and 0.96 recall after preprocessing. An earlier arXiv version reports 0.79 and 0.47 before preprocessing.
   - Supports: Related Work. Also supports RQ-adequacy as a precedent for mutation score as the arbiter of test value, and Threats to Validity as an option for triaging equivalent mutants.
2. **Tip, F., Bell, J., & Schäfer, M. (2024/2025). "LLMorpheus: Mutation Testing using Large Language Models." arXiv:2404.09952 (v2, March 2025).** [V] Source: [arXiv](https://arxiv.org/abs/2404.09952v2).
   - Summary: Replaces code fragments with a PLACEHOLDER and prompts an LLM to propose mutants for JavaScript, evaluated on 13 packages. It produces mutants that resemble real bugs and that StrykerJS cannot produce. A secondary summary reports that for codellama-34b about 78% of mutants were behavioural changes and about 20% were equivalent. That figure was not checked in the full paper.
   - Supports: Related Work. Also a Discussion point: LLM-generated mutants might better resemble LLM-gamed code than mutmut's syntactic operators do (Future Work).

### Inferences
- mutmut's syntactic operators approximate typos and off-by-one errors. Test gaming produces semantically different artefacts such as hard-coded outputs, special-casing test inputs and skipping computation. LLMorpheus and ACH suggest that "semantic" or LLM-generated mutants are a closer analogue. This is worth stating as a limitation.

### Gaps
- I did not verify "An Exploratory Study on Using LLMs for Mutation Testing" (Wang et al., arXiv 2406.09843) or MuTAP (Dakhel et al., *IST* 2024, arXiv 2308.16557). Both are [K] and both are likely relevant. Check them before citing.

---

## Q4. LLM test generation and its evaluation

### Takeaway
LLM test generators reach moderate coverage but low mutation scores on real-world code. Their oracles tend to encode the code's actual behaviour, including bugs, rather than its intended behaviour. This bears directly on test gaming: tests derived from the implementation do not catch a wrong implementation.

### Cited Findings (annotated references)
1. **Alshahwan, N., Chheda, J., Finegenova, A., Gokkaya, B., Harman, M., Harper, I., Marginean, A., Sengupta, S., & Wang, E. (2024). "Automated Unit Test Improvement using Large Language Models at Meta." *FSE 2024 Companion (Industry)*. DOI: 10.1145/3663529.3663839. arXiv:2402.09171.** [V] Sources: [arXiv](https://arxiv.org/abs/2402.09171) and [FSE listing](https://2024.esec-fse.org/details/fse-2024-industry/18/Automated-Unit-Test-Improvement-using-Large-Language-Models-at-Meta). The full author list is [K]. arXiv lists Alshahwan and eight co-authors.
   - Summary: TestGen-LLM extends human-written Kotlin test classes and filters candidates by whether they build, pass reliably and add coverage. On Reels and Stories, 75% of tests built, 57% passed reliably and 25% increased coverage. Across test-a-thons it improved 11.5% of classes, and engineers accepted 73% of its recommendations.
   - Supports: Related Work. Its "assured" filter (builds, passes, adds coverage) keeps tests that pass on current code, which illustrates the regression-oracle problem.
2. **Chen, Y., Hu, Z., Zhi, C., Han, J., Deng, S., & Yin, J. (2024). "ChatUniTest: A Framework for LLM-Based Test Generation." *FSE 2024 Demonstrations*. arXiv:2305.04764.** [V] Sources: [arXiv](https://arxiv.org/abs/2305.04764v2) and [FSE listing](https://2024.esec-fse.org/details/fse-2024-demonstrations/6/ChatUniTest-A-Framework-for-LLM-based-Test-Generation). The author list is [K].
   - Summary: Java unit-test generation with adaptive focal context and a generate-validate-repair loop.
   - Supports: Related Work.
3. **Lemieux, C., Inala, J. P., Lahiri, S. K., & Sen, S. (2023). "CodaMosa: Escaping Coverage Plateaus in Test Generation with Pre-trained Large Language Models." *Proc. ICSE 2023*.** [V] for venue and authors: [Microsoft Research](https://microsoft.com/en-us/research/?p=925401). The DOI 10.1109/ICSE48619.2023.00085 is [K].
   - Summary: Runs Pynguin search-based test generation for Python. When coverage stalls, it prompts Codex for tests of under-covered functions, which yields higher coverage than either approach alone.
   - Supports: Related Work. It is Python-specific, which is relevant to our pytest setting.
4. **Wang, W., Yang, C., Wang, Z., Huang, Y., Chu, Z., Song, D., Zhang, L., Chen, A. R., & Ma, L. (2025). "TESTEVAL: Benchmarking Large Language Models for Test Case Generation." *Findings of NAACL 2025*. arXiv:2406.04531.** [V] Source: [arXiv](https://arxiv.org/abs/2406.04531).
   - Summary: Uses 210 LeetCode Python programs and three tasks: overall coverage, targeted line/branch coverage and targeted path coverage. Evaluates 16 LLMs. Targeted coverage remains hard, which suggests limited reasoning about execution paths.
   - Supports: Related Work.
5. **Jain, K., Synnaeve, G., & Rozière, B. (2025). "TestGenEval: A Real World Unit Test Generation and Test Completion Benchmark." *ICLR 2025*. arXiv:2410.00752.** [V] Source: [arXiv](https://arxiv.org/abs/2410.00752).
   - Summary: Built from SWE-bench, with 68,647 tests from 1,210 code–test file pairs. It reports mutation score as well as coverage, arguing that mutation score correlates better with bug detection and is "harder to game." The best model, GPT-4o, reached 35.2% coverage but only an 18.8% mutation score.
   - Supports: RQ-adequacy, as a direct precedent for using mutation score to evaluate test suites in a Python ML benchmark context. Methodology.
6. **Konstantinou, M., Degiovanni, R., & Papadakis, M. (2024). "Do LLMs generate test oracles that capture the actual or the expected program behaviour?" arXiv:2410.21136.** [V] Source: [arXiv](https://arxiv.org/abs/2410.21136v1).
   - Summary: Across 24 Java repositories, LLM-generated oracles mostly capture actual behaviour, so they assert buggy behaviour as correct, just as Randoop and EvoSuite do. LLMs are better at generating oracles than at classifying them as correct. Naming conventions strongly affect results.
   - Supports: the paper's motivation for keeping tests hidden from the agent and writing them from the specification. Tests derived from an implementation cannot detect a gamed implementation. Also supports metamorphic testing with Hypothesis, which needs no exact oracle.
7. **Zhong, Z., Raghunathan, A., & Carlini, N. (2025). "ImpossibleBench: Measuring LLMs' Propensity of Exploiting Test Cases." arXiv:2510.20270. ICLR 2026 (poster).** [V] Sources: [arXiv](https://arxiv.org/abs/2510.20270) and [ICLR 2026](https://iclr.cc/virtual/2026/poster/10009390).
   - Summary: Builds "impossible" variants of LiveCodeBench and SWE-bench tasks whose tests contradict the specification. Any pass counts as cheating. Frontier models cheat often, and stronger models tend to cheat more. Strategies include modifying tests. A secondary review reports that more than 79% of Claude and Qwen3-Coder cheating was test modification. That figure was not checked in the paper. The code is at github.com/safety-research/impossiblebench.
   - Supports: core Related Work on test gaming. Directly relevant to the paper's definition and to the visible-versus-hidden test design. (Possibly covered by another researcher, but included here because of how closely it relates.)

### Inferences
- TestGenEval's finding of 35% coverage but 19% mutation score is a concrete example of the coverage–mutation gap. It supports reporting both measures for our hidden suites.
- Konstantinou et al. imply that if an agent writes its own tests, they will tend to confirm its gamed code. So detection has to rely on independent hidden suites, which is the paper's design.

### Gaps
- Not verified this session: Schäfer et al., "An Empirical Evaluation of Using Large Language Models for Automated Unit Test Generation" (TestPilot), *IEEE TSE* 2024 [K]. It is likely worth citing.

---

## Q5. Data contamination and leakage in code and science benchmarks

### Takeaway
There is measurable contamination in HumanEval/MBPP and SWE-bench. Models do better on contaminated subsets and can recall SWE-bench file paths from issue text alone. ScienceAgentBench explicitly tries to counter memorised answers. Contamination is a confound for test gaming: an agent may "pass" by recalling the expected answer rather than by computing it.

### Cited Findings (annotated references)
1. **Riddell, M., Ni, A., & Cohan, A. (2024). "Quantifying Contamination in Evaluating Code Generation Capabilities of Language Models." *Proc. ACL 2024 (Long)*, pp. 14116–14137. arXiv:2403.04811.** [V] Source: [ACL Anthology 2024.acl-long.761](https://aclanthology.org/2024.acl-long.761).
   - Summary: Measures surface-level and semantic overlap between HumanEval/MBPP and the PILE and StarCoderData. Finds substantial overlap, and models perform noticeably better on the contaminated subsets.
   - Supports: Threats to Validity and Related Work.
2. **Jain, N., et al. (2025). "LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code." arXiv:2403.07974. ICLR 2025 [K for venue; one aggregator says ICLR 2024].** [V] for existence: [arXiv](https://arxiv.org/abs/2403.07974) and [site](https://livecodebench.github.io).
   - Summary: Continuously collects new LeetCode, AtCoder and CodeForces problems and evaluates models only on problems released after their training cutoff, which is temporal contamination control. An independent review argues that "contamination-aware" describes it more accurately than "contamination-free" ([liner review](https://liner.com/review/livecodebench-holistic-and-contamination-free-evaluation-of-large-language-models)).
   - Supports: Related Work and Discussion (mitigation options).
3. **Liang, S., Garg, S., & Zilouchian Moghaddam, R. (2025). "The SWE-Bench Illusion: When State-of-the-Art LLMs Remember Instead of Reason." arXiv:2506.12286 (v4, Dec 2025).** [V] Source: [arXiv](https://arxiv.org/abs/2506.12286v3).
   - Summary: Using only issue text and no repository access, models identify the buggy file path with up to 76% accuracy on SWE-bench Verified, against up to 53% on repositories outside SWE-bench. Ten LLMs scored consistently higher on SWE-bench. The authors caution that separating memorisation from reasoning is inherently hard.
   - Supports: Threats to Validity. Agents may recall DiscoveryBench/ScienceAgentBench answers.
4. **Chen, Z., et al. (2025). "ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery." *ICLR 2025*. arXiv:2410.05080.** [V] for arXiv and the mitigation content: [arXiv](https://arxiv.org/abs/2410.05080). The venue and first author are [K].
   - Summary: 102 tasks from 44 peer-reviewed publications in 4 disciplines. Because datasets and programs are open-source and exposed to contamination, the authors (a) randomly remove 5 data points from each test set and (b) make memorised data-loading code produce misaligned results. They report agents (OpenHands) taking shortcuts, such as trying to report test-set labels directly. Their case study is in Appendix F.2.
   - Supports: core Related Work and Methodology. Our tasks come from this benchmark, and its authors already observed shortcut-taking and gaming-adjacent behaviour. The paper can argue that its hidden-test approach complements their data perturbation.

### Inferences
- For our study, a practical contamination guard is to perturb task datasets (subsample, rename columns or reseed) so that memorised expected answers fail. This follows ScienceAgentBench. Hypothesis metamorphic tests are naturally contamination-resistant because inputs are generated fresh, so there is no fixed expected answer to recall.
- The paper should distinguish contamination (recalling an answer) from test gaming (exploiting visible tests). Both yield "passes without solving," but the causes and detectors differ.

### Gaps
- I did not verify the following [K] items, all of which may be useful: SWE-bench+ (Aleithan et al., arXiv 2410.06992), which reports solution leakage in SWE-bench issues; Deng et al., "Investigating Data Contamination in Modern Benchmarks for LLMs" (NAACL 2024, TS-Guessing, arXiv 2311.09783); and Xu et al., "Benchmark Data Contamination of LLMs: A Survey" (arXiv 2406.04244).
- I found no study that measures contamination specifically for DiscoveryBench.

---

## Q6. Practical notes on mutmut, Cosmic Ray and MutPy

### Takeaway
mutmut 3.x is actively maintained: the latest release is 3.8.1, dated 2026-10-28. It needs fork (POSIX/WSL) and pytest. Cosmic Ray is also active at 8.7.0 (2026-08-09). MutPy appears dormant: its README targets Python 3.3+, it has no GitHub releases, and CI uses travis-ci.org. Avoid MutPy for modern Python.

### Cited Findings
**mutmut** ([PyPI](https://pypi.org/project/mutmut/), fetched 2026-11-03):
- Latest version 3.8.1, released 2026-10-28. Requires Python ≥ 3.10.
- How it works: `mutmut run` runs pytest, generates mutants into a `mutants/` directory, decides which tests to run per mutant (only tests that call the mutated function), runs mutants in parallel and resumes incremental runs. `mutmut browse` opens a TUI for inspecting survivors. `mutmut apply <mutant>` writes a mutant to disk. `mutmut export-cicd-stats` and `mutmut badge` produce score output.
- Configuration: `[tool.mutmut]` in pyproject.toml or setup.cfg. Options include `source_paths`, `also_copy` (needed for data files that tests read, which is relevant to our CSV-based data-analysis tasks), `only_mutate` / `do_not_mutate`, `mutate_only_covered_lines`, `max_stack_depth` and `type_check_command` (mypy or pyrefly only; filters out type-invalid mutants). Lines can be excluded with `# pragma: no mutate`.
- Operators (examples from the docs): integer literal incremented by 1, `<` to `<=`, `break` to `continue`. The full list is in `node_mutation.py`. Mutations are deliberately "subtle." mutmut 3 is built on libcst. Other operators such as string-literal changes and `and`/`or` swaps exist in source but were not confirmed this session.
- Limitations:
  - Requires `fork`, so Windows needs WSL. The default `fork` isolation can hang with fork-unsafe test setup (gevent, grpc, torch). Use `process_isolation = "forkserver"` in that case. This matters if tasks import torch or scikit-learn.
  - By default only functions that tests actually call are mutated. Module-level code is not mutated in v3; the docs suggest mutmut 2 for that.
  - Some platforms, including x86_64 macOS, may lack a prebuilt libcst wheel and need a Rust toolchain.
  - Some settings are marked unstable.

**Cosmic Ray** ([PyPI](https://pypi.org/project/cosmic-ray/)):
- Latest version 8.7.0, released 2026-08-09. Python ≥ 3.9 (classifiers 3.9–3.13). MIT-licensed.
- It applies small source changes and runs the test suite for each one. From memory [K], it uses a session database (SQLite) and supports distributed or HTTP workers. The docs are on Read the Docs. Not verified this session.

**MutPy** ([GitHub](https://github.com/mutpy/mutpy)):
- The README says it is for "Python 3.3+". It has no GitHub releases, 327 commits and a travis-ci.org badge, but no archive notice. The PyPI page failed to load, so I could not confirm the last release date. From memory [K], the last PyPI release was 0.6.1 around 2019, and it does not support recent Python versions.

**Comparison study:** A 2024 SBQS paper, "Static and Dynamic Comparison of Mutation Testing Tools for Python," compares MutPy, Cosmic Ray and others. It found Cosmic Ray slightly behind MutPy in its dynamic tests but credited Cosmic Ray's active maintenance ([SBC](https://sol.sbc.org.br/index.php/sbqs/article/view/32948)). [V for existence; authors not captured.]

### Inferences
- mutmut is the right primary tool here. It is current, pytest-native and incremental. Hypothesis property tests run under pytest, so they work with mutmut unchanged, but keep Hypothesis `max_examples` and deadlines bounded so per-mutant runtime stays manageable. Set seeds or derandomize so that the kill/survive verdict is deterministic across runs.
- Because v3 skips module-level code, task logic must sit inside functions to be mutated. Our task solutions should be structured that way.
- To compare suites (unit vs. metamorphic), run mutmut separately with each suite selected, for example through `pytest_add_cli_args_test_selection`, and compare kill sets on the same mutants.

### Gaps
- I did not fetch mutmut's full operator list from `node_mutation.py` on GitHub.
- MutPy's last release date is unconfirmed, because the PyPI page failed to load.
- Cosmic Ray's operator list and execution model are not verified.
