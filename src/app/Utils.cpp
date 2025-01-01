#include "Utils.h"

using namespace std;

string Utils::base64Encode(const vector<uint8_t> &input) {
    string ret;
    int i = 0;
    int j = 0;
    uint8_t tempCharArray3[3];
    uint8_t tempCharArray4[4];

    int inputLength = static_cast<int>(input.size());
    int index = 0;

    while (inputLength--) {
        tempCharArray3[i++] = input.at(index++);
        if (i == 3) {
            tempCharArray4[0] = (tempCharArray3[0] & 0xfc) >> 2;
            tempCharArray4[1] = ((tempCharArray3[0] & 0x03) << 4) + ((tempCharArray3[1] & 0xf0) >> 4);
            tempCharArray4[2] = ((tempCharArray3[1] & 0x0f) << 2) + ((tempCharArray3[2] & 0xc0) >> 6);
            tempCharArray4[3] = tempCharArray3[2] & 0x3f;

            for (i = 0; (i < 4); i++)
                ret += kEncodeLookup[tempCharArray4[i]];
            i = 0;
        }
    }

    if (i) {
        for (j = i; j < 3; j++)
            tempCharArray3[j] = '\0';

        tempCharArray4[0] = (tempCharArray3[0] & 0xfc) >> 2;
        tempCharArray4[1] = ((tempCharArray3[0] & 0x03) << 4) + ((tempCharArray3[1] & 0xf0) >> 4);
        tempCharArray4[2] = ((tempCharArray3[1] & 0x0f) << 2) + ((tempCharArray3[2] & 0xc0) >> 6);

        for (j = 0; (j < i + 1); j++)
            ret += kEncodeLookup[tempCharArray4[j]];

        while ((i++ < 3))
            ret += '=';

    }

    return ret;
}

string Utils::encodeImgToBase64(const cv::Mat &img, int quality) {
    vector<uint8_t> encodedImage;
    cv::imencode(Constants::JPG_EXTENSION, img, encodedImage, {cv::IMWRITE_JPEG_QUALITY, quality});
    return Utils::base64Encode(encodedImage);
}

string Utils::dateTimeToStr(time_t dateTime, bool oldFormat) {
    char dateTimeString[26];
    time(&dateTime);
    struct tm *ptime;
    ptime = gmtime(&dateTime);
    ptime->tm_hour = (ptime->tm_hour + 6) % 24;
    if(oldFormat)
        strftime(dateTimeString, sizeof dateTimeString, "%F %T", ptime);
    else
        strftime(dateTimeString, sizeof dateTimeString, "%FT%TZ", ptime);
    return string(dateTimeString);
}

string Utils::pointToStr(float x, float y) {
    return to_string(x) + ", " + to_string(y);
}

vector<string> Utils::splitString(const string &str, const string &delimiter) {
    size_t cur = 0;
    size_t next;
    vector<string> subStrings;
    while ((next = str.find(delimiter, cur)) != string::npos) {
        auto subString = str.substr(cur, next - cur);
        subStrings.push_back(move(subString));
        cur = next + delimiter.length();
    }
    subStrings.push_back(str.substr(cur));
    return subStrings;
}

int Utils::calculateEditDistance(const string &str1, const string &str2) {
    if (str1.empty() || str2.empty()) {
        return (int)max(str2.size(), str1.size());
    }

    int n = (int)str1.size() + 1;
    int m = (int)str2.size() + 1;
    vector<int> distances(static_cast<int>(n * m), 0);

    for (int i = 1; i < n; i++) { distances[i * m] = i; }
    for (int j = 1; j < m; j++) { distances[j] = j; }

    for (int i = 1; i < n; i++) {
        for (int j = 1; j < m; j++) {
            int substitution = str1[i - 1] == str2[j - 1] ? 0 : 1;
            distances[i * m + j] = min(distances[(i - 1) * m + j] + 1,
                                       min(distances[i * m + j - 1] + 1,
                                           distances[(i - 1) * m + j - 1] + substitution));
        }
    }
    return distances[n * m - 1];
}

double Utils::calculateDistanceBetweenTwoPoints(const cv::Point2f &point1, const cv::Point2f &point2) {
    return sqrt(pow(point1.x - point2.x, 2) + pow(point1.y - point2.y, 2));
}

std::string Utils::dateTimeToStrAnother(time_t dateTime) {
    char dateTimeString[30];
    time(&dateTime);
    struct tm *ptime;
    ptime = gmtime(&dateTime);
    ptime->tm_hour = (ptime->tm_hour + 6) % 24;

    strftime(dateTimeString, sizeof dateTimeString, "%Y-%m-%d_%H:%M:%S.000", ptime);
    return string(dateTimeString);
}

