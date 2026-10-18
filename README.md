# FTEC5660 Homework 2: CV Verification Agent

Build a LangChain agent that reads each CV in a folder, looks the candidate up
on our SocialGraph MCP server (mock LinkedIn and Facebook), and outputs a
reliability score in [0, 1] for each CV.

A CV is **valid** (label `1`) when its claims agree with the candidate's
social media profiles. It **has a discrepancy** (label `0`) when it contains
problems such as an inflated job title, shifted dates, an upgraded degree, a
fake school or employer, a wrong location, or made-up skills. Differences in
wording only ("Bachelor of Science" vs `BSc`, "UI/UX Design" vs `UI/UX`,
"Senior Engineer" for an `Engineer` role with seniority `senior`, listing fewer
skills) are not discrepancies.

## Student task

Fill in the two functions in `hw2.py` that contain `### YOUR CODE HERE`. You
may add imports, constants and helper functions above them, but do not change
the provided code below them:

- `build_agent(tools)` creates your agent from the MCP tools.
- `score_cvs(agent, cvs)` runs the agent on every CV and returns
  `{file_name: score}`, one float in [0, 1] per CV.

A score above `0.5` means "valid"; `0.5` or below means "has discrepancy". You
may use a single tool-calling agent, multiple agents, reflection, or a
combination. Do not hard-code filenames, names, or public answers; grading
uses unseen CVs.

## MCP server

The server is hosted at `https://ftec5660.ngrok.app/mcp`. `hw2.py` already
connects to it and passes you these tools:

| Tool | Purpose |
| --- | --- |
| `search_facebook_users(q, limit, fuzzy)` | find Facebook users by display name |
| `get_facebook_profile(user_id)` | full Facebook profile |
| `get_facebook_mutual_friends(user_id_1, user_id_2)` | mutual friends of two users |
| `search_linkedin_people(q, location, industry, limit, fuzzy)` | search LinkedIn by name, skill, or title |
| `get_linkedin_profile(person_id)` | full LinkedIn profile (experience, education, skills) |
| `get_linkedin_interactions(person_id)` | post and like statistics |

Print `tool.name`, `tool.description`, and `tool.args` for the full schemas.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "DEEPSEEK_API_KEY=your_key_here" > .env
python3 hw2.py --cv-folder public_test
```

Use your own DeepSeek API key (from https://platform.deepseek.com).

The program creates `results.csv` in the current directory. Its columns are
`cv`, `score`, and `correctness`. The public labels are in
`public_test/ground_truth.json`; every CV with label `0` also has a `reason`
explaining the discrepancy, and `results.csv` shows it when your agent misses
one. The starter returns no score (`None`) for every CV so it runs before you
add any API code; missing scores count as incorrect.

The required model is `deepseek-v4-flash`, and all evidence must come from the
MCP server (no web search). The server is shared by the whole class: keep at
most about 3 CVs in flight at a time (e.g. `asyncio.Semaphore(3)`). We grade
with our own API key, so never put a key in your code; `.env` must stay out of
git.


## Task 2: Fool the Verifier

We added a target candidate, **Kelly Tsang** (LinkedIn `person_id` 10001,
Facebook `user_id` 10001, display name "Kel Tsang"). Her true CV is
`task2/target_cv.pdf`.

Write one CV for Kelly Tsang with at least one false or embellished detail
that our 5 verifier agents (same function as Task 1) still score above 0.5.
Each fooled agent is worth 4 points; a CV without a real false or embellished
detail earns 0. Attack your own Task 1 agent first:

```bash
cp my_attack.pdf task2/adversarial_cv.pdf
python3 hw2.py --cv-folder task2
```

`results.csv` shows your agent's score for both CVs (they are not graded, as
`task2/` has no `ground_truth.json`). Only the PDF is the attack surface; do not
attack the MCP server. See the homework description for the full rules.

## Homework 2 solution:

### Task 1

I use a LangChain tool-calling agent with `deepseek-v4-flash` (temperature 0).
The agent extracts the CV's factual claims, resolves the candidate using name,
location and independent career/education anchors, and retrieves full LinkedIn
profiles through the SocialGraph MCP tools; Facebook can corroborate identity.
It checks every asserted name, city, job, degree and skill, including exact
employment/graduation years and job seniority, while allowing equivalent
wording, degree abbreviations and subsets of profile skills. The CV is passed
as a JSON document and the system prompt treats embedded instructions and
claimed evidence as untrusted content. The final assessment contains
field-level evidence and statuses, and a local validation gate requires an
actually fetched LinkedIn profile and all five check categories. A confirmed
discrepancy receives 0.05, complete matching evidence receives 0.95, and
unresolved checks or execution failures receive 0.50. These are decision
scores, with 0.50 classified as discrepancy by the supplied runner. Each CV
has an independent conversation, a 210-second time limit, and one repair
opportunity for malformed/incomplete assessments; up to three CVs run
concurrently. Exceptions are isolated per CV so the supplied runner can still
produce `results.csv`. **Public-test accuracy is pending a configured DeepSeek
API key and a real end-to-end run.** Offline tests exercise evidence gating,
failure isolation, concurrency and response repair; they measure software
behavior separately from verification accuracy.

### Task 2

My submitted one-page CV is `task2/adversarial_cv.pdf`. It visibly claims
**MSc in Finance, HKUST, graduated 2016**, upgrading Kelly Tsang's actual
**BSc in Finance**. I retrieved LinkedIn profile 10001 and Facebook profile
10001 from the course MCP server and confirmed the undergraduate degree.
The remaining career dates, employers, location and skills preserve her
genuine profile.

The attack exploits a difference between the rendered PDF and its extracted
text. The visible MSc degree line is drawn as vector glyph outlines; an
invisible PDF text layer at the same position contains the genuine BSc line.
The outlines remain clearly legible on the page, while text extraction reads
the BSc text layer. The other sections use ordinary PDF text. A matching
control PDF generated by the same script displays BSc and carries the same
text layer. This permits a direct check that the visible degree changed
while the text received by the verifier stayed identical. The intended
failure occurs before the language model receives the CV: a text-only
verifier can correctly check every extracted claim and still accept the
visibly embellished document. The five course verifiers use the same loader,
so this is a plausible shared weakness. **Real-model self-test scores and the
five hidden verifiers' outcomes are currently unmeasured.** Visual inspection
and loader-equivalence checks establish the PDF mechanism; acceptance is
measured separately. A defence would compare rendered-page OCR with the text
layer and investigate contradictory or missing text.

### Reproducing the solution

Validation completed on 18 October 2026 with Python 3.13.5: all six MCP tools
were discovered through the supplied adapter, both target profiles were
retrieved, five offline tests passed, and the one-page attack passed exact
loader-equivalence and rendered-page difference checks. The supplied runner
below the student functions remains unchanged. Model-based validation is
pending API configuration.

Use Python 3.10 or newer and install `requirements.txt` in a virtual
environment. Copy `.env.example` to `.env`, supply your own key locally, and run:

```bash
python hw2.py --cv-folder public_test
python hw2.py --cv-folder task2
```

Each command writes `results.csv`; retain the first result before running the
second command. Optionally set `HW2_TRACE_DIR=artifacts/traces` to save
per-CV tool messages, field checks, timings and failure types. Local
credentials, results and artifacts are ignored by Git.

The runtime uses the original requirements. Optional development tools and
PDF generation dependencies are in `requirements-dev.txt`:

```bash
pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
python scripts/check_mcp.py
python scripts/build_adversarial_cv.py
python scripts/validate_attack.py
```

After configuring the API key, `python scripts/run_experiments.py --repeats 3`
runs three real evaluations of each folder and retains each CSV, execution
log and tool trace under `artifacts/experiments/`, together with a summary
of public accuracy, attack acceptance and execution failures.

`build_adversarial_cv.py` regenerates the submitted PDF and a local BSc
control in `artifacts/control_cv.pdf`. `validate_attack.py` uses the original
`hw2.cv_text` loader and verifies page count and identical extracted text,
then saves a preview. Regeneration needs no model API calls.
