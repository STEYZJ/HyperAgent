import unittest
from pathlib import Path


class ThirdPartyNoticesTest(unittest.TestCase):
    def test_notice_file_mentions_direct_runtime_dependencies(self):
        root = Path(__file__).resolve().parents[1]
        text = (root / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")

        for name in [
            "NumPy",
            "SciPy",
            "scikit-learn",
            "Matplotlib",
            "PyYAML",
            "PyTorch",
            "tifffile",
        ]:
            self.assertIn(name, text)

    def test_notice_file_marks_reference_projects_as_non_vendored(self):
        root = Path(__file__).resolve().parents[1]
        text = (root / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")

        self.assertIn("Design reference only", text)
        self.assertIn("参考/", text)
        self.assertIn("ignored by git", text)

    def test_readmes_link_to_notice_file(self):
        root = Path(__file__).resolve().parents[1]

        for path in [root / "README.md", root / "README.zh-CN.md"]:
            self.assertIn(
                "THIRD_PARTY_NOTICES.md",
                path.read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
