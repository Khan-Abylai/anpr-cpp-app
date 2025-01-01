#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include "Constants.h"

class TemplateMatching {
public:
    explicit TemplateMatching(std::unordered_map<std::string, Constants::CountryCode> templates);

private:

    const std::vector<std::vector<std::string>> SQUARE_TEMPLATES_HALF_KZ{{"999", "99AAA"},
                                                                         {"999", "99AA"},
                                                                         {"99",  "99AA"},
                                                                         {"A99", "9999"}};


    const std::unordered_map<Constants::CountryCode, std::string> COUNTRY_TO_STRING{
        {Constants::CountryCode::MA, "MA"},
        {Constants::CountryCode::TA, "TA"}
    };

    std::unordered_map<std::string, Constants::CountryCode> COUNTRY_TEMPLATES{
                {"AA99AAA",   Constants::CountryCode::MA},
                {"AA9999A", Constants::CountryCode::MA},
                {"AAA9999", Constants::CountryCode::MA},
                {"AA9999", Constants::CountryCode::MA},
                {"AAAAA", Constants::CountryCode::MA},
                {"AAA9999A", Constants::CountryCode::MA},
                {"AAA999", Constants::CountryCode::MA},
                {"AAA99", Constants::CountryCode::MA},
                {"AAA9", Constants::CountryCode::MA},
                {"AAA999A", Constants::CountryCode::MA},
                {"AA999A", Constants::CountryCode::MA},
                {"AA99A", Constants::CountryCode::MA},
                {"AAA", Constants::CountryCode::MA},
                {"AA999", Constants::CountryCode::MA},
                {"AA9", Constants::CountryCode::MA},
                {"AA99", Constants::CountryCode::MA},
                {"A999", Constants::CountryCode::MA},
                {"AA99999", Constants::CountryCode::MA},
                {"A9999", Constants::CountryCode::MA},
                {"A9999A", Constants::CountryCode::MA},
                {"AA9A", Constants::CountryCode::MA},
                {"AA99999A", Constants::CountryCode::MA},
                {"AA9999AA", Constants::CountryCode::MA},
                {"AA9999AA", Constants::CountryCode::MA},
                {"999999", Constants::CountryCode::TA}
    };

    const char STANDARD_DIGIT = '9';
    const char STANDARD_ALPHA = 'A';

    std::string standardizeLicensePlate(const std::string &plateLabel) const;

public:
    std::string processSquareLicensePlate(const std::string &topPlateLabel, const std::string &bottomPlateLabel);

    std::string getCountryCode(const std::string &plateLabel);
};
