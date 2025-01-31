import logging
import pytest
from main import extract_answer_options, find_correct_answer

# Configure logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


@pytest.mark.parametrize("query, expected_options, gpt_response, expected_answer", [
    # Test Case 1
    (
        "В каком городе находится главный кампус Университета ИТМО?\n"
        "1. Москва\n2. Санкт-Петербург\n3. Екатеринбург\n4. Нижний Новгород\n",
        ["Москва", "Санкт-Петербург", "Екатеринбург", "Нижний Новгород"],
        "Из информации на сайте",
        2
    ),
    # Test Case 2
    (
        "В каком году Университет ИТМО был включён в число Национальных исследовательских университетов России?\n"
        "1. 2007\n2. 2009\n3. 2011\n4. 2015",
        ["2007", "2009", "2011", "2015"],
        "Университет ИТМО был включён в число Национальных исследовательских университетов России в 2009 году.",
        2
    ),
    # Test Case 3
    (
        "В каком рейтинге (по состоянию на 2021 год) ИТМО впервые вошёл в топ-400 мировых университетов?\n"
        "1. ARWU (Shanghai Ranking)\n"
        "2. Times Higher Education (THE) World University Rankings\n"
        "3. QS World University Rankings\n"
        "4. U.S. News & World Report Best Global Universities",
        [
            "ARWU (Shanghai Ranking)",
            "Times Higher Education (THE) World University Rankings",
            "QS World University Rankings",
            "U.S. News & World Report Best Global Universities"
        ],
        "ИТМО впервые вошёл в топ-400 мировых университетов в рейтинге QS World University Rankings.",
        3
    )
])
def test_answer_extraction(caplog, query, expected_options, gpt_response, expected_answer):
    extracted_options = extract_answer_options(query)
    caplog.set_level(logging.WARNING)
    logging.info(f"Extracted options: {extracted_options}")

    assert extracted_options == expected_options, "Extracted answer options do not match expected values"

    answer = find_correct_answer(gpt_response, extracted_options)
    logging.info(f"GPT Response: {gpt_response}")
    logging.info(f"Detected Answer: {answer}")

    assert answer == expected_answer, f"Expected answer {expected_answer}, but got {answer}"
