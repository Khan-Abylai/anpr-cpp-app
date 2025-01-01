//
// Created by kartykbayev on 12/5/23.
//
#pragma once

#include <iostream>
#include <string>
#include <fstream>
#include <nlohmann/json.hpp>
#include "app/CameraScope.h"
#include "app/Utils.h"
#include "app/Constants.h"

class Template {
public:
    static bool parseJson(const std::string &fileName);

    static std::unordered_map<std::string, Constants::CountryCode> getTemplateMap();
};
