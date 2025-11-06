from utils import trim_diff

def test_trim_diff():
    patch = """@@ -1,2 +1,2 @@
- old line
+ new line
"""
    out = trim_diff(patch)
    assert "new line" in out
    assert "old line" in out
