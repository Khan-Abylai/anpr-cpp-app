#include "CalibParams.h"

using namespace std;
using json = nlohmann::json;

CalibParams::CalibParams(const string &serverIp, const string &cameraIp,
                         pair<float, float> calibrationSizes) : ILogger("Calib Params " + cameraIp),
                                                                cameraIp(cameraIp) {

    CALIBRATION_FRAME_WIDTH = calibrationSizes.first;
    CALIBRATION_FRAME_HEIGHT = calibrationSizes.second;
    calibParamsUrl = serverIp + cameraIp;
    LOG_INFO("Calibration Parameter Url : %s", calibParamsUrl.c_str());
    getMask();
}

void CalibParams::getMask() {
    auto responseText = sendRequestForMaskPoints();
    auto polygonPoints = getPolygonPoints(responseText, "mask2");
    auto subPolygonPoints = getPolygonPoints(responseText, "mask");

    lock_guard<mutex> guard(maskAccessChangeMutex);
    mask = cv::Mat::zeros((int) CALIBRATION_FRAME_HEIGHT, (int) CALIBRATION_FRAME_WIDTH, CV_8UC1);
    mask2 = cv::Mat::zeros((int) CALIBRATION_FRAME_HEIGHT, (int) CALIBRATION_FRAME_WIDTH, CV_8UC1);

    cv::fillConvexPoly(mask, polygonPoints, WHITE_COLOR);
    cv::fillConvexPoly(mask2, subPolygonPoints, WHITE_COLOR);
}

vector<cv::Point2i> CalibParams::getPolygonPoints(const string &polygonPointsStr, const string &maskType) const {
    vector<cv::Point2i> polygonPoints;
    if (polygonPointsStr.empty() || polygonPointsStr.length() <= 2) {
        polygonPoints = getDefaultPolygonPoints();
    } else {
        auto polygonPointsJson = json::parse(polygonPointsStr);
        for (auto &point: polygonPointsJson[maskType])
            polygonPoints.emplace_back(point["x"].get<int>(), point["y"].get<int>());
        if (polygonPoints.empty())
            polygonPoints = getDefaultPolygonPoints();
    }
    return polygonPoints;
}

string CalibParams::sendRequestForMaskPoints() {
    cpr::Response response = cpr::Get(cpr::Url{calibParamsUrl}, cpr::VerifySsl(false), cpr::Timeout(timeout));
    if (response.status_code >= 400 || response.status_code == 0) {
        LOG_ERROR("%s Error [%d] making request for mask", cameraIp.data(), response.status_code);
        return "";
    }
    return response.text;
}

cv::Point2i CalibParams::getRelatedPoint(const cv::Point2f &point, const cv::Size &imageSize) const {
    auto xPoint = (point.x * CALIBRATION_FRAME_WIDTH) / (float) imageSize.width;
    auto yPoint = (point.y * CALIBRATION_FRAME_HEIGHT) / (float) imageSize.height;
    return cv::Point2i{(int) xPoint, (int) yPoint};
}

bool CalibParams::isPointInTheMask(const cv::Point2i &point) {
    lock_guard<mutex> guard(maskAccessChangeMutex);
    return mask.at<uchar>(point.y, point.x) == WHITE_COLOR;
}

bool
CalibParams::isLicensePlateInSelectedArea(const shared_ptr<LicensePlate> &licensePlate, const std::string &maskType) {
    cv::Point2f centerPointInGround{static_cast<float>(licensePlate->getCenter().x), (float) (licensePlate->getCenter().y)};

    auto centerPointHd = getRelatedPoint(centerPointInGround, licensePlate->getCarImageSize());

    if (maskType == "main")
        return isPointInTheMask(centerPointHd);
    else
        return isPointInTheSubMask(centerPointHd);
}

void CalibParams::showCenterPointInGround(const shared_ptr<LicensePlate> &licensePlate,
                                          const cv::Point2f &centerPointInGround) {
    cv::Mat copyImage;
    licensePlate->getCarImage().copyTo(copyImage);
    cv::circle(copyImage, centerPointInGround, 3, cv::Scalar(255, 0, 0), cv::FILLED);

    cv::Mat displayImage;
    cv::resize(copyImage, displayImage,cv::Size(copyImage.cols * 0.7,copyImage.rows * 0.7), 0, 0);
    cv::imshow(licensePlate->getCameraIp(), displayImage);
    cv::waitKey(60);
}

const string &CalibParams::getCameraIp() const {
    return cameraIp;
}

bool CalibParams::isPointInTheSubMask(const cv::Point2i &point) {
    lock_guard<mutex> guard(maskAccessChangeMutex);
    return mask2.at<uchar>(point.y, point.x) == WHITE_COLOR;
}

std::vector<cv::Point2i> CalibParams::getDefaultPolygonPoints() const {
    vector<cv::Point2i> polygonPoints;

    polygonPoints.emplace_back(0, 0);
    polygonPoints.emplace_back(static_cast<int>(CALIBRATION_FRAME_WIDTH), 0);
    polygonPoints.emplace_back(static_cast<int>(CALIBRATION_FRAME_WIDTH), static_cast<int>(CALIBRATION_FRAME_HEIGHT));
    polygonPoints.emplace_back(0, static_cast<int>(CALIBRATION_FRAME_HEIGHT));

    return polygonPoints;
}



