#include "TemplateMatching.h"
#include "Constants.h"

using namespace std;

TemplateMatching::TemplateMatching(unordered_map<std::string, Constants::CountryCode> templates) {
    if (!templates.empty())
        COUNTRY_TEMPLATES = std::move(templates);
}

string TemplateMatching::processSquareLicensePlate(const string &topPlateLabel,
                                                   const string &bottomPlateLabel) {

    string standardizedTopLabel = standardizeLicensePlate(topPlateLabel);
    string standardizedBottomLabel = standardizeLicensePlate(bottomPlateLabel);

    for (auto &half_templates: SQUARE_TEMPLATES_HALF_KZ) {
        if (half_templates[0] == standardizedTopLabel &&
            half_templates[1] == standardizedBottomLabel) {
            return topPlateLabel + bottomPlateLabel.substr(2) + bottomPlateLabel.substr(0, 2);
        }
    }
    return std::move(topPlateLabel + bottomPlateLabel);
}

string TemplateMatching::getCountryCode(const string &plateLabel) {
    auto country = COUNTRY_TEMPLATES.find(standardizeLicensePlate(plateLabel));

    if (country == COUNTRY_TEMPLATES.end()) {
        return Constants::UNIDENTIFIED_COUNTRY;
    }
    return COUNTRY_TO_STRING.at(country->second);
}

string TemplateMatching::standardizeLicensePlate(const string &plateLabel) const {
    string standardizedPlateLabel = plateLabel;

    for (auto charIndex = 0; charIndex < plateLabel.length(); charIndex++) {
        if (isdigit(plateLabel[charIndex])) {
            standardizedPlateLabel[charIndex] = STANDARD_DIGIT;
        } else if (isalpha(plateLabel[charIndex])) {
            standardizedPlateLabel[charIndex] = STANDARD_ALPHA;
        }
    }
    return std::move(standardizedPlateLabel);
}


