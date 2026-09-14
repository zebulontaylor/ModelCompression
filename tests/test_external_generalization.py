from scripts.evaluate_external_generalization import (
    exact_mcnemar_p,
    gsm8k_target,
    parse_arc,
    parse_last_number,
)


def test_arc_parser_accepts_leading_answer_letter():
    assert parse_arc("C. oxygen") == "C"
    assert parse_arc("Answer: b") == "B"
    assert parse_arc("no choice given") == ""


def test_gsm8k_parser_prefers_explicit_final():
    assert parse_last_number("2 + 3 = 5. FINAL: 5") == "5"
    assert parse_last_number("FINAL: \\boxed{1,024}") == "1024"
    assert parse_last_number("The intermediate values are 3 and 4") == "4"
    assert gsm8k_target("working\n#### 1,024") == "1024"


def test_exact_mcnemar_p():
    assert exact_mcnemar_p(0, 0) == 1.0
    assert exact_mcnemar_p(1, 0) == 1.0
    assert exact_mcnemar_p(10, 0) == 2 / 2**10
