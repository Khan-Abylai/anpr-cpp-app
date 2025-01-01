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
            {Constants::CountryCode::KZ, "KZ"},
            {Constants::CountryCode::KG, "KG"},
            {Constants::CountryCode::UZ, "UZ"},
            {Constants::CountryCode::RU, "RU"},
            {Constants::CountryCode::BY, "BY"},
            {Constants::CountryCode::GE, "GE"},
            {Constants::CountryCode::AM, "AM"},
            {Constants::CountryCode::AZ, "AZ"},
    };

    std::unordered_map<std::string, Constants::CountryCode> COUNTRY_TEMPLATES{
            {"999AAA99",  Constants::CountryCode::KZ},
            {"999AA99",   Constants::CountryCode::KZ},
            {"99AA99",    Constants::CountryCode::KZ},
            {"999AAA",    Constants::CountryCode::KZ},
            {"A999AAA",   Constants::CountryCode::KZ},
            {"A999AA",    Constants::CountryCode::KZ},
            {"A999AA99",  Constants::CountryCode::RU},
            {"AA99999",   Constants::CountryCode::RU},
            {"A999AA999", Constants::CountryCode::RU},
            {"AA999A99",  Constants::CountryCode::RU},
            {"999A99999", Constants::CountryCode::RU},
            {"99999AAA",  Constants::CountryCode::KG},
            {"99999AA",   Constants::CountryCode::KG},
            {"99A999AA",  Constants::CountryCode::UZ},
            {"A999999",   Constants::CountryCode::KZ},
            {"AAA9999",   Constants::CountryCode::KZ},
            {"AA999",     Constants::CountryCode::KZ},
            {"999999",    Constants::CountryCode::KZ},
            {"999999A",   Constants::CountryCode::KG},
            {"AAAA9999",  Constants::CountryCode::KG},
            {"99AA999",   Constants::CountryCode::AM},
            {"9999AA9",   Constants::CountryCode::BY},
            {"AA999AA",   Constants::CountryCode::GE},
            {"AA999AAA",  Constants::CountryCode::GE},
            {"9999AAA",   Constants::CountryCode::KG},
            {"9999AA",    Constants::CountryCode::KG},
            {"999AA",     Constants::CountryCode::KG},
            {"A9999AA",   Constants::CountryCode::KG},
            {"A9999A",    Constants::CountryCode::KG},
            {"9999AA",    Constants::CountryCode::KZ},
            {"AAA999",    Constants::CountryCode::UZ},
            {"AA9999",    Constants::CountryCode::UZ},
            {"99A999999", Constants::CountryCode::UZ},
            {"9999AA99",  Constants::CountryCode::UZ},
            {"A99999999", Constants::CountryCode::UZ},
            {"99999A9A",  Constants::CountryCode::KG},
            {"99999999",  Constants::CountryCode::KG},
            {"AAA99999",  Constants::CountryCode::RU},
            {"AAA999A",   Constants::CountryCode::KG},
            {"AA9999AA",  Constants::CountryCode::KG},
            {"999A999",   Constants::CountryCode::AZ},
            {"999AA999",  Constants::CountryCode::AZ},
            {"999AAA999", Constants::CountryCode::AZ},
            {"99AAA999",  Constants::CountryCode::AZ},
            {"99AAAA99",  Constants::CountryCode::AZ},
            {"A9",        Constants::CountryCode::AZ},
            {"AA999999",  Constants::CountryCode::AZ},
            {"999AAA9",   Constants::CountryCode::AZ},
            {"AAA9",      Constants::CountryCode::AZ},


    };

    const char STANDARD_DIGIT = '9';
    const char STANDARD_ALPHA = 'A';

    std::string standardizeLicensePlate(const std::string &plateLabel) const;

public:
    std::string processSquareLicensePlate(const std::string &topPlateLabel, const std::string &bottomPlateLabel);

    std::string getCountryCode(const std::string &plateLabel);
};
