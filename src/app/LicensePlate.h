#pragma once

#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <vector>

#include "Constants.h"
#include "Utils.h"
#include "../ITimer.h"

class LicensePlate : public ITimer {
public:

    /**
 * @class LicensePlate
 *
 * @brief Represents a license plate with its positioning information and detection probability.
 *
 * The LicensePlate class encapsulates the necessary data to represent a license plate, including its position
 * (x, y, width, height) and the coordinates of its four corners (x1, y1, x2, y2, x3, y3, x4, y4). In addition, it
 * stores the detection probability of the license plate.
 *
 * The LicensePlate class is used to pass license plate information across different components of an application,
 * such as license plate recognition systems or image processing algorithms.
 */

    LicensePlate(int x, int y, int w, int h, int x1, int y1, int x2, int y2, int x3, int y3, int x4, int y4, float detProb);

    /**
 * @brief Retrieves the center point of the object.
 *
 * This function returns the center point of the object as a reference, ensuring that the
 * original object is not modified. The returned center point is constant, meaning it cannot
 * be modified through the returned reference.
 *
 * @return A constant reference to the center point of the object.
 */

    [[nodiscard]] const cv::Point2i &getCenter() const;

    [[nodiscard]] const cv::Point2f &getLeftTop() const;

    [[nodiscard]] const cv::Point2f &getRightBottom() const;

    [[nodiscard]] const cv::Point2f &getLeftBottom() const;

    [[nodiscard]] const cv::Point2f &getRightTop() const;

    [[nodiscard]] bool isSquare() const;

    [[nodiscard]] float getArea() const;

    [[nodiscard]] int getWidth() const;

    [[nodiscard]] int getHeight() const;

    [[nodiscard]] cv::Size getCarImageSize() const;

    [[nodiscard]] const cv::Mat &getPlateImage() const;

    void setPlateImage(const cv::Mat &frame);

    [[nodiscard]] const std::string &getPlateLabel() const;

    void setLicensePlateLabel(std::string lpLabel);

    [[nodiscard]] const std::string &getCameraIp() const;

    void setCameraIp(std::string ip);

    [[nodiscard]] const cv::Mat &getCarImage() const;

    void setCarImage(cv::Mat image);

    void setRTPtimestamp(double timestamp);

    [[nodiscard]] double getRTPtimestamp() const;

    void setCarModel(std::string carModelParam);

    [[nodiscard]] const std::string &getCarModel() const;

    void setCarModelProb(double carModelProbability);

    [[nodiscard]] double getCarModelProb() const;

    void setDirection(std::string direction);

    [[nodiscard]] const std::string &getDirection() const;

    [[nodiscard]] const std::string &getResultSendUrl() const;

    void setResultSendUrl(const std::string &url);

    [[nodiscard]] const std::string &getSecondaryResultSendUrl() const;

    void setSecondaryResultSendUrl(const std::string &url);

    [[nodiscard]] bool doesSecondaryUrlEnabled() const;

    void setSecondaryUrlEnabledFlag(bool flag);

    [[nodiscard]] bool doesCarBBoxDefined() const;

    void setCarBBoxFlag(bool flag);

    void setVehicleImage(cv::Mat frame);

    [[nodiscard]] const cv::Mat &getVehicleImage() const;

    [[nodiscard]] double getSharpness() const ;

    [[nodiscard]] double getDFTSharpness() const ;

    [[nodiscard]] double getQuality() const ;

    [[nodiscard]] double getWhiteness() const;

    void setPlateProbability(double probability);

    [[nodiscard]] double getPlateProbability() const;

    void setRealTimeOfEvent(double time);

    double getRealTimeOfEvent();

private:
    static double calculateLaplacian(const cv::Mat &imageCrop);

    static double calculateWhiteScore(const cv::Mat &imageCrop);

    static double calculateQualityMetric(double laplacianValue, double whiteScore);

    static double calculateSharpness(const cv::Mat &licensePlateImg);

    static double calculateDFTSharpness(const cv::Mat &image);

    static double calculateBlurCoefficient(const cv::Mat &image);

    const int CROP_PADDING = 3;
    const float SQUARE_LP_RATIO = 2.6;
    cv::Mat plateImage;
    cv::Mat carImage;
    cv::Mat vehicleImage;
    cv::Mat croppedPlateImage;
    std::string licensePlateLabel;
    std::string carModel;
    double carModelProb;
    std::string cameraIp;
    std::string direction;

    std::string resultSendUrl, secondaryResultSendUrl;
    bool secondarySendUrlEnabled{};
    bool bboxCarEnabled = false;

    double rtpTimestamp{};
    double realTimeOfPackage;
    cv::Point2i center;
    cv::Point2f leftTop;
    cv::Point2f leftBottom;
    cv::Point2f rightTop;
    cv::Point2f rightBottom;
    int width, height;
    bool square = false;

    double laplacianValue{};
    double dftValue{};
    double qualityValue{};
    double whitenessValue{};
    double plateProbability;
};