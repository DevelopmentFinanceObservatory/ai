"""`op_spec_hash` and toolkit `content_version` minting: determinism, domain
separation, duplicate refusal, and the recompute path a replayer uses.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from analysis_mcp.engine.filter import run_filter
from analysis_mcp.execution_identity import ExecutionIdentity, current_execution_identity
from analysis_mcp.identity import (
    OP_SPEC_HASH_PATTERN,
    TOOLKIT_CONTENT_VERSION_PATTERN,
    OpSpecIdentityError,
    build_op_spec_preimage,
    canonicalize_op_spec,
    mint_toolkit_content_version,
    op_spec_hash_for,
    recompute_op_spec_hash,
    recompute_toolkit_content_version,
)
from analysis_mcp.op_spec import (
    AggregateOpSpec,
    CategoryMatchPredicate,
    OpSpecAdapter,
    QuotientOfOpSpec,
    ShareOfOpSpec,
)
from common.materialisation import FrameRef
from common.recipe import normal_form
from dfo_frame_grammar import Frame
from pydantic import ValidationError

_FIXTURES = Path(__file__).parent / "fixtures"

SNAPSHOT_REF = {
    "spec_id": "spec_" + "a" * 32,
    "content_version": "cv_" + "b" * 32,
}


def _op_spec(**overrides: object) -> AggregateOpSpec:
    return AggregateOpSpec(
        frame_ref=FrameRef(**SNAPSHOT_REF),
        group_by=overrides.get("group_by", ()),
        aggregation=overrides.get("aggregation", "sum"),
    )


def test_op_spec_hash_has_the_published_shape() -> None:
    op_spec_hash = op_spec_hash_for(_op_spec())
    assert OP_SPEC_HASH_PATTERN.fullmatch(op_spec_hash)


def test_op_spec_hash_is_deterministic() -> None:
    assert op_spec_hash_for(_op_spec(group_by=("reporter",))) == op_spec_hash_for(
        _op_spec(group_by=("reporter",))
    )


def test_op_spec_hash_is_independent_of_group_by_order() -> None:
    a = op_spec_hash_for(_op_spec(group_by=("reporter", "year")))
    b = op_spec_hash_for(_op_spec(group_by=("year", "reporter")))
    assert a == b


def test_op_spec_hash_moves_when_the_op_spec_changes() -> None:
    a = op_spec_hash_for(_op_spec(group_by=("reporter",)))
    b = op_spec_hash_for(_op_spec(group_by=("year",)))
    assert a != b


def test_nfd_category_match_matches_the_same_rows_as_its_nfc_form() -> None:
    """`Frame.build` NFC-normalises every cell it stores, and
    `CategoryMatchPredicate.values` normalises to NFC at construction too --
    so an NFD-spelled predicate matches the identical rows its NFC form
    does, rather than minting two different results against an NFC-stored
    cell.
    """
    nfd = "café"
    nfc = "café"
    frame = Frame.build(
        role_keys=["reporter"],
        rows=[
            {"stat_var": "dcid:X", "reporter": nfc, "time_period": "2020", "value": 1.0},
            {"stat_var": "dcid:X", "reporter": "dcid:OTHER", "time_period": "2020", "value": 2.0},
        ],
    )
    nfd_result = run_filter(
        frame, predicates=(CategoryMatchPredicate(column="reporter", values=(nfd,)),)
    )
    nfc_result = run_filter(
        frame, predicates=(CategoryMatchPredicate(column="reporter", values=(nfc,)),)
    )
    assert nfd_result.kept_row_count == nfc_result.kept_row_count == 1
    assert nfd_result.frame == nfc_result.frame


def test_duplicate_group_by_column_refuses() -> None:
    with pytest.raises(OpSpecIdentityError) as exc_info:
        op_spec_hash_for(_op_spec(group_by=("reporter", "reporter")))
    assert exc_info.value.category == "duplicate_member"


_TWO_SIDED_OP_SPECS = pytest.mark.parametrize("model", [ShareOfOpSpec, QuotientOfOpSpec])


def _two_sided_op_spec(
    model: type[ShareOfOpSpec | QuotientOfOpSpec], **overrides: object
) -> ShareOfOpSpec | QuotientOfOpSpec:
    return model(
        frame_ref=FrameRef(**SNAPSHOT_REF),
        numerator_by=overrides.get("numerator_by", ()),
        denominator_by=overrides.get("denominator_by", {}),
    )


@_TWO_SIDED_OP_SPECS
def test_op_spec_hash_is_independent_of_numerator_by_order(
    model: type[ShareOfOpSpec | QuotientOfOpSpec],
) -> None:
    a = op_spec_hash_for(_two_sided_op_spec(model, numerator_by=("donor", "sector")))
    b = op_spec_hash_for(_two_sided_op_spec(model, numerator_by=("sector", "donor")))
    assert a == b


def test_share_of_and_quotient_of_over_the_same_parameters_mint_different_hashes() -> None:
    assert op_spec_hash_for(_two_sided_op_spec(ShareOfOpSpec)) != op_spec_hash_for(
        _two_sided_op_spec(QuotientOfOpSpec)
    )


@_TWO_SIDED_OP_SPECS
def test_op_spec_hash_is_independent_of_denominator_by_key_order(
    model: type[ShareOfOpSpec | QuotientOfOpSpec],
) -> None:
    """Two calls differing only in `denominator_by`'s own key insertion
    order mint one `op_spec_hash` and (via `frame_ref_resolution.
    canonical_entity_binding`) execute over one order -- JCS sorts an
    object's own members regardless of Python dict insertion order, so this
    requires no explicit registration in `identity._normalise_parameters`,
    unlike an array-shaped field."""
    a = op_spec_hash_for(
        _two_sided_op_spec(
            model,
            numerator_by=("donor", "sector"),
            denominator_by={"donor": "provider", "sector": "area"},
        )
    )
    b = op_spec_hash_for(
        _two_sided_op_spec(
            model,
            numerator_by=("sector", "donor"),
            denominator_by={"sector": "area", "donor": "provider"},
        )
    )
    assert a == b


def test_recompute_op_spec_hash_reproduces_the_mint_from_canonical_bytes() -> None:
    op_spec = _op_spec(group_by=("reporter",))
    minted = op_spec_hash_for(op_spec)
    canonical_bytes = canonicalize_op_spec(op_spec)
    assert recompute_op_spec_hash(canonical_bytes) == minted


def test_recompute_op_spec_hash_detects_a_tampered_byte() -> None:
    op_spec = _op_spec(group_by=("reporter",))
    minted = op_spec_hash_for(op_spec)
    tampered = canonicalize_op_spec(op_spec).replace(b"reporter", b"reportee")
    assert recompute_op_spec_hash(tampered) != minted


def test_toolkit_content_version_has_the_published_shape() -> None:
    op_spec_hash = op_spec_hash_for(_op_spec(group_by=("reporter",)))
    content_version = mint_toolkit_content_version(
        op_spec_hash=op_spec_hash,
        input_content_versions=[SNAPSHOT_REF["content_version"]],
        execution_identity=current_execution_identity(),
    )
    assert TOOLKIT_CONTENT_VERSION_PATTERN.fullmatch(content_version)


def test_toolkit_content_version_is_independent_of_input_order() -> None:
    op_spec_hash = op_spec_hash_for(_op_spec(group_by=("reporter",)))
    other_cv = "cv_" + "c" * 32
    ei = current_execution_identity()
    a = mint_toolkit_content_version(
        op_spec_hash=op_spec_hash,
        input_content_versions=[SNAPSHOT_REF["content_version"], other_cv],
        execution_identity=ei,
    )
    b = mint_toolkit_content_version(
        op_spec_hash=op_spec_hash,
        input_content_versions=[other_cv, SNAPSHOT_REF["content_version"]],
        execution_identity=ei,
    )
    assert a == b


def test_toolkit_content_version_moves_with_execution_identity() -> None:
    op_spec_hash = op_spec_hash_for(_op_spec(group_by=("reporter",)))
    a = mint_toolkit_content_version(
        op_spec_hash=op_spec_hash,
        input_content_versions=[SNAPSHOT_REF["content_version"]],
        execution_identity=ExecutionIdentity(
            engine_version="0.1.0", frame_grammar_version="1.0.0", reduction_policy_version="0.1.0"
        ),
    )
    b = mint_toolkit_content_version(
        op_spec_hash=op_spec_hash,
        input_content_versions=[SNAPSHOT_REF["content_version"]],
        execution_identity=ExecutionIdentity(
            engine_version="0.2.0", frame_grammar_version="1.0.0", reduction_policy_version="0.1.0"
        ),
    )
    assert a != b


def test_duplicate_input_content_version_refuses() -> None:
    op_spec_hash = op_spec_hash_for(_op_spec(group_by=("reporter",)))
    with pytest.raises(OpSpecIdentityError) as exc_info:
        mint_toolkit_content_version(
            op_spec_hash=op_spec_hash,
            input_content_versions=[
                SNAPSHOT_REF["content_version"],
                SNAPSHOT_REF["content_version"],
            ],
            execution_identity=current_execution_identity(),
        )
    assert exc_info.value.category == "duplicate_member"


def test_recompute_toolkit_content_version_matches_mint() -> None:
    op_spec_hash = op_spec_hash_for(_op_spec(group_by=("reporter",)))
    ei = current_execution_identity()
    minted = mint_toolkit_content_version(
        op_spec_hash=op_spec_hash,
        input_content_versions=[SNAPSHOT_REF["content_version"]],
        execution_identity=ei,
    )
    recomputed = recompute_toolkit_content_version(
        op_spec_hash=op_spec_hash,
        input_content_versions=[SNAPSHOT_REF["content_version"]],
        execution_identity=ei,
    )
    assert recomputed == minted


class TestOpSpecHashVectors:
    """The `ops1` golden vector suite: ~12 positive and negative cases."""

    _vectors = json.loads((_FIXTURES / "vectors" / "op_spec_hash_vectors.json").read_text())[
        "vectors"
    ]

    @pytest.mark.parametrize("vector", _vectors, ids=lambda v: v["name"])
    def test_vector(self, vector: dict[str, object]) -> None:
        op_spec_dict = vector["op_spec"]
        if vector["classification"] == "positive":
            op_spec = OpSpecAdapter.validate_python(op_spec_dict)
            assert op_spec_hash_for(op_spec) == vector["expected_op_spec_hash"]
        else:
            with pytest.raises((ValidationError, OpSpecIdentityError)):
                op_spec = OpSpecAdapter.validate_python(op_spec_dict)
                op_spec_hash_for(op_spec)


_ROLLING_VECTORS = json.loads(
    (_FIXTURES / "vectors" / "op_spec_rolling_hash_vectors.json").read_text()
)["vectors"]
_ROLLING_POSITIVE = {v["name"]: v for v in _ROLLING_VECTORS if v["classification"] == "positive"}


class TestRollingOpSpecHashVectors:
    """The `ops1` vectors for the rolling overlay, each beside its fixed twin."""

    @pytest.mark.parametrize("vector", _ROLLING_VECTORS, ids=lambda v: v["name"])
    def test_vector(self, vector: dict[str, object]) -> None:
        if vector["classification"] == "positive":
            op_spec = OpSpecAdapter.validate_python(vector["op_spec"])
            assert canonicalize_op_spec(op_spec).decode() == vector["expected_canonical"]
            assert op_spec_hash_for(op_spec) == vector["expected_op_spec_hash"]
            assert (
                recompute_op_spec_hash(vector["expected_canonical"].encode())
                == (vector["expected_op_spec_hash"])
            )
        else:
            with pytest.raises(ValidationError):
                OpSpecAdapter.validate_python(vector["op_spec"])

    def test_a_population_stating_no_level_hashes_as_one_without_the_member(self) -> None:
        vector = _ROLLING_POSITIVE["growth-rolling-end"]
        stated = json.loads(json.dumps(vector["op_spec"]))
        for population in stated["rolling"]["population"]:
            population["levels"] = {}
        levelled = _ROLLING_POSITIVE["growth-rolling-end-country-level"]

        op_spec = OpSpecAdapter.validate_python(stated)

        assert canonicalize_op_spec(op_spec).decode() == vector["expected_canonical"]
        assert levelled["expected_op_spec_hash"] != vector["expected_op_spec_hash"]
        assert '"levels":{"observationAbout_type":"Country"}' in levelled["expected_canonical"]

    def test_only_the_respelled_population_shares_a_hash(self) -> None:
        names_by_hash: dict[str, set[str]] = {}
        for name, vector in _ROLLING_POSITIVE.items():
            names_by_hash.setdefault(vector["expected_op_spec_hash"], set()).add(name)

        assert [names for names in names_by_hash.values() if len(names) > 1] == [
            {"growth-rolling-end-two-roots", "growth-rolling-population-respelled"}
        ]

    @pytest.mark.parametrize(
        ("name", "moved"),
        [
            ("growth-rolling-end", {"window_end": "2025"}),
            ("growth-rolling-both-ends", {"window_start": "2020", "window_end": "2025"}),
            ("rank-rolling-window", {"window_start": "2020", "window_end": "2025"}),
        ],
    )
    def test_a_rolling_window_keeps_its_recipe_normal_form_when_the_latest_period_moves(
        self, name: str, moved: dict[str, str]
    ) -> None:
        op_spec = _ROLLING_POSITIVE[name]["op_spec"]
        twin = {key: value for key, value in op_spec.items() if key != "rolling"}

        assert _normal_form_of({**op_spec, **moved}) == _normal_form_of(op_spec)
        assert _normal_form_of({**twin, **moved}) != _normal_form_of(twin)

    def test_a_rolling_predicate_keeps_its_recipe_normal_form_when_the_latest_period_moves(
        self,
    ) -> None:
        def at(vector: str, period: str) -> dict[str, object]:
            op_spec = _ROLLING_POSITIVE[vector]["op_spec"]
            predicates = [
                {**predicate, "value": period}
                if predicate["column"] == "time_period"
                else predicate
                for predicate in op_spec["predicates"]
            ]
            return {**op_spec, "predicates": predicates}

        rolling = "filter-one-rolling-predicate"
        assert _normal_form_of(at(rolling, "2020")) == _normal_form_of(at(rolling, "2019"))
        fixed = "filter-fixed-predicates"
        assert _normal_form_of(at(fixed, "2020")) != _normal_form_of(at(fixed, "2019"))


def _normal_form_of(op_spec: dict[str, object]) -> dict[str, object]:
    """The recipe normal form of one verb call over a dataset root."""
    canonical = json.loads(canonicalize_op_spec(OpSpecAdapter.validate_python(op_spec)))
    tree = {"op_spec": canonical, "inputs": {"frame_ref": {"root": 0}}}
    return normal_form(tree, [{"dataset": {"spec_id": "spec_" + "a" * 32}, "options": {}}])


def test_an_unset_rank_limit_stays_out_of_the_preimage_and_a_set_one_enters() -> None:
    unlimited = OpSpecAdapter.validate_python(
        {"verb": "rank", "frame_ref": SNAPSHOT_REF, "rank_by": ["reporter"]}
    )
    limited = OpSpecAdapter.validate_python(
        {"verb": "rank", "frame_ref": SNAPSHOT_REF, "rank_by": ["reporter"], "limit": 3}
    )

    assert "limit" not in build_op_spec_preimage(unlimited)["parameters"]
    assert build_op_spec_preimage(limited)["parameters"]["limit"] == 3
    assert op_spec_hash_for(limited) != op_spec_hash_for(unlimited)


class TestToolkitContentVersionVectors:
    """The `opcv1` golden vector suite."""

    _vectors = json.loads(
        (_FIXTURES / "vectors" / "toolkit_content_version_vectors.json").read_text()
    )["vectors"]

    @pytest.mark.parametrize("vector", _vectors, ids=lambda v: v["name"])
    def test_vector(self, vector: dict[str, object]) -> None:
        input_ = vector["input"]
        ei = ExecutionIdentity(**input_["execution_identity"])
        if vector["classification"] == "positive":
            content_version = mint_toolkit_content_version(
                op_spec_hash=input_["op_spec_hash"],
                input_content_versions=input_["input_content_versions"],
                execution_identity=ei,
            )
            assert content_version == vector["expected_content_version"]
        else:
            with pytest.raises(OpSpecIdentityError):
                mint_toolkit_content_version(
                    op_spec_hash=input_["op_spec_hash"],
                    input_content_versions=input_["input_content_versions"],
                    execution_identity=ei,
                )
