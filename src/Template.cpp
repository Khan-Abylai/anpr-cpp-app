//
// Created by kartykbayev on 12/5/23.
//

#include "Template.h"

using namespace std;
using namespace nlohmann;
std::unordered_map<std::string, Constants::CountryCode> countryTemplates;

bool Template::parseJson(const std::string &fileName) {
    try {
        ifstream templateFile(fileName);

        if (!templateFile.is_open())
            throw runtime_error("Config file not found");

        json templates = json::parse(templateFile);

        if (templates.find("templates") == templates.end()) {
            return false;
        }

        auto templateList = templates["templates"];
        for (const auto &item: templateList.items()) {
            auto key = item.key();
            auto value = Constants::STRING_TO_COUNTRY.at(item.value().get<string>());
            countryTemplates.insert(std::make_pair(key, value));
        }
    } catch (exception &e) {
        return false;
    }
    return true;
}

std::unordered_map<std::string, Constants::CountryCode> Template::getTemplateMap() {
    return countryTemplates;
}
