import numpy2 as np2


def test_compat_report_shape():
    r = np2.compat.report()
    assert isinstance(r, dict)
    assert r["package"] == "numpy2"
    assert isinstance(r.get("subset"), dict)
    assert set(["mgrid", "ogrid", "c_", "r_"]).issubset(r["subset"].keys())
    assert isinstance(r.get("not_implemented"), list)


def test_compat_report_detects_known_stubs():
    r = np2.compat.report()
    names = {item.get("qualified_name") for item in r["not_implemented"]}
    assert "numpy2._MGridClass.__getitem__" in names
    assert "numpy2.array.mgrid_func" in names
    assert "numpy2.array.ogrid_func" in names
