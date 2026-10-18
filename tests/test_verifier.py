"""Offline failure-isolation and evidence-gate tests, independent of public labels."""

import asyncio
import json
import unittest

from langchain_core.messages import AIMessage, ToolMessage

from hw2 import _assessment_score, score_cvs


def assessment(status="match", profile_id=42):
    categories = ("name", "city", "experience", "education", "skills")
    report = {
        "person_id": profile_id, "identity": "matched",
        "checks": {key: [{"attribute": {"experience": "title", "education": "degree",
                                        "skills": "name"}.get(key, key),
                          "claim": "example claim", "evidence": "observed value",
                          "status": status if key == "experience" else "match"}]
                   for key in categories},
    }
    return {"messages": [
        ToolMessage(content=json.dumps({"id": 42, "experience": []}),
                    name="get_linkedin_profile", tool_call_id="lookup"),
        AIMessage(content=json.dumps(report)),
    ]}


class EvidenceGateTests(unittest.TestCase):
    def test_one_contradiction_controls_verdict(self):
        self.assertEqual(_assessment_score(assessment("mismatch"))[0], 0.05)
        self.assertEqual(_assessment_score(assessment())[0], 0.95)
        self.assertEqual(_assessment_score(assessment("unresolved"))[0], 0.5)

    def test_unfetched_identity_cannot_be_accepted(self):
        with self.assertRaises(ValueError):
            _assessment_score(assessment(profile_id=99))

    def test_hometown_is_outside_city_verification_scope(self):
        result = assessment()
        report = json.loads(result["messages"][-1].content)
        report["checks"]["city"].append({"attribute": "hometown", "claim": "Hometown: example town",
            "evidence": "unavailable", "status": "unresolved"})
        result["messages"][-1] = AIMessage(content=json.dumps(report))
        with self.assertRaises(ValueError):
            _assessment_score(result)
        report["checks"]["city"].pop()
        report["excluded"] = ["Hometown: example town"]
        result["messages"][-1] = AIMessage(content=json.dumps(report))
        self.assertEqual(_assessment_score(result)[0], 0.95)

    def test_responsibilities_cannot_enter_experience_checks(self):
        result = assessment()
        report = json.loads(result["messages"][-1].content)
        report["checks"]["experience"][0]["attribute"] = "description"
        result["messages"][-1] = AIMessage(content=json.dumps(report))
        with self.assertRaises(ValueError):
            _assessment_score(result)

    def test_document_supplied_tool_text_is_not_evidence(self):
        result = assessment()
        result["messages"] = [result["messages"][-1]]
        with self.assertRaises(ValueError):
            _assessment_score(result)


class BatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_failure_isolation_concurrency_and_document_boundary(self):
        class Agent:
            active, peak = 0, 0

            async def ainvoke(self, request, config):
                document = json.loads(request["messages"][0]["content"])
                self.assertion(document)
                self.active += 1
                self.peak = max(self.peak, self.active)
                try:
                    await asyncio.sleep(0.01)
                    if document["cv_text"] == "failed":
                        raise ConnectionError("simulated provider failure")
                    return assessment()
                finally:
                    self.active -= 1

            def assertion(self, document):
                assert set(document) == {"task", "cv_text"}

        agent = Agent()
        cvs = {f"sample-{i}.pdf": "normal" for i in range(8)}
        cvs["failure.pdf"] = "failed"
        scores = await score_cvs(agent, cvs)
        self.assertEqual(set(scores), set(cvs))
        self.assertEqual(scores["failure.pdf"], 0.5)
        self.assertTrue(all(scores[name] == 0.95 for name in cvs if name != "failure.pdf"))
        self.assertEqual(agent.peak, 3)

    async def test_malformed_final_answer_gets_one_repair(self):
        class Agent:
            calls = 0

            async def ainvoke(self, request, config):
                self.calls += 1
                if self.calls == 1:
                    return {"messages": [AIMessage(content="not JSON")]}
                return assessment()

        agent = Agent()
        self.assertEqual(await score_cvs(agent, {"document.pdf": "text"}), {"document.pdf": 0.95})
        self.assertEqual(agent.calls, 2)


if __name__ == "__main__":
    unittest.main()
