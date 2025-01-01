#include "LPRecognizerService.h"

using namespace std;


LPRecognizerService::LPRecognizerService(const shared_ptr<SharedQueue<std::shared_ptr<Package>>> &packageQueue,
                                         const vector<CameraScope> &cameras, float recognizerThreshold,
                                         pair<float, float> calibrationSizes, int statInterval,
                                         const std::unordered_map<std::string, Constants::CountryCode> &templates)
        : ILogger(
        "LP Recognizer Service "),
          RECOGNIZER_THRESHOLD{
                  recognizerThreshold},
          cameraScopeVector{
                  cameras},
          waitMinutes{
                  statInterval} {
    initializeCameraIpCarTrackerMap(packageQueue, cameras, calibrationSizes);

    lpQueue = make_unique<SharedQueue<shared_ptr<LicensePlate>>>();
    lpRecognizer = make_unique<LPRecognizer>();
    templateMatching = make_unique<TemplateMatching>(templates);
    backgroundThread = thread(&LPRecognizerService::checkCurrentPlatesStatus, this);
    for (const auto &camera: cameras) {
        cameraToNotValidByPatternPlatesCount.insert({camera.getCameraIp(), 0});
        cameraToNotValidByConfPlatesCount.insert({camera.getCameraIp(), 0});
        cameraToNotValidBothPlatesCount.insert({camera.getCameraIp(), 0});

        cameraToOverallPlatesCount.insert({camera.getCameraIp(), 0});
        cameraToPassPlatesCount.insert({camera.getCameraIp(), 0});
        cameraIpToCurrentCameraWhiteness.insert({camera.getCameraIp(), 0});
        cameraIpToPlateWhitenessSum.insert({camera.getCameraIp(), 0});
        cameraIpToPlateWhitenessNum.insert({camera.getCameraIp(), 0});
        cameraIpToPlateDFTSharpnessSum.insert({camera.getCameraIp(), 0});
        cameraIpToPlateLaplacianSharpnessSum.insert({camera.getCameraIp(), 0});
        cameraIpToWhitePlatesNum.insert({camera.getCameraIp(), 0});
        cameraIpToDarkPlatesNum.insert({camera.getCameraIp(), 0});

        cameraToNotValidProbs.insert({camera.getCameraIp(), {}});
        cameraToValidProbs.insert({camera.getCameraIp(), {}});
    }
}

void LPRecognizerService::initializeCameraIpCarTrackerMap(
        const shared_ptr<SharedQueue<std::shared_ptr<Package>>> &packageQueue, const vector<CameraScope> &cameras,
        pair<float, float> calibrationSizes) {
    for (const auto &camera: cameras) {
        unique_ptr<BaseCarTracker> carTracker;

        auto calibParams = make_shared<CalibParams>(camera.getNodeIp(), camera.getCameraIp(), calibrationSizes);
        carTracker = make_unique<MaskCarTracker>(packageQueue, std::move(calibParams), camera);
        carTracker->run();
        cameraIpToCarTrackerMap.insert({camera.getCameraIp(), std::move(carTracker)});
    }
}

void LPRecognizerService::addToQueue(shared_ptr<LicensePlate> licensePlate) {
    lpQueue->push(std::move(licensePlate));
}

bool LPRecognizerService::isValidLicensePlate(const string &lpLabel, float probability) {
    auto stop = 1;
    auto plateCountry = templateMatching->getCountryCode(lpLabel);
    auto isTemplateMatched = plateCountry != Constants::UNIDENTIFIED_COUNTRY;
    return probability > RECOGNIZER_THRESHOLD && isTemplateMatched;
}

pair<string, float>
LPRecognizerService::getLicensePlateLabel(const vector<pair<string, float>> &recognizerResult, bool isSquarePlate) {
    float probability;
    string licensePlateLabel;
    if (isSquarePlate) {
        string label = recognizerResult.front().first;
        string sep = ".";
        vector<string> parts = string_split(label, '.');

        if (parts.size() == 2) {
            licensePlateLabel = templateMatching->processSquareLicensePlate(parts[0], parts[1]);
        } else {
            licensePlateLabel = label;
        }
        probability = recognizerResult.front().second;
    } else {
        licensePlateLabel = recognizerResult.front().first;
        probability = recognizerResult.front().second;
    }
    return make_pair(licensePlateLabel, probability);
}


void LPRecognizerService::addPlateToTrack(const shared_ptr<LicensePlate> &licensePlate) {
    string cameraIp = licensePlate->getCameraIp();
    auto event = cameraIpToCarTrackerMap.find(cameraIp);
    if (event != cameraIpToCarTrackerMap.end())
        event->second->track(licensePlate);
}


pair<cv::Mat, cv::Mat> LPRecognizerService::divideImageIntoHalf(const cv::Mat &lpImage) {
    lpImage(cv::Rect(0, 0, Constants::SQUARE_LP_W, Constants::SQUARE_LP_H / 2)).copyTo(
            whiteImage.colRange(0, Constants::SQUARE_LP_W).rowRange(0, Constants::RECT_LP_H));
    auto firstHalf = whiteImage.clone();

    lpImage(cv::Rect(0, Constants::SQUARE_LP_H / 2, Constants::SQUARE_LP_W, Constants::SQUARE_LP_H / 2)).copyTo(
            whiteImage.colRange(0, Constants::SQUARE_LP_W).rowRange(0, Constants::RECT_LP_H));
    auto secondHalf = whiteImage.clone();

    return make_pair(firstHalf, secondHalf);
}

vector<cv::Mat> LPRecognizerService::getLicensePlateImages(const shared_ptr<LicensePlate> &licensePlate) {
    vector<cv::Mat> lpImages;
    if (licensePlate->isSquare()) {
        auto combinedLicensePlate = combineInOneLineSquareLpPlate(licensePlate->getPlateImage());
        lpImages.push_back(combinedLicensePlate);
    } else {
        lpImages.push_back(licensePlate->getPlateImage());
    }
    return lpImages;
}

void LPRecognizerService::run() {
    while (!shutdownFlag) {
        auto licensePlate = lpQueue->wait_and_pop();
        if (licensePlate == nullptr) continue;
        vector<cv::Mat> lpImages = getLicensePlateImages(licensePlate);
        auto startTime = chrono::high_resolution_clock::now();
        auto recognizerResult = lpRecognizer->predict(lpImages);
        auto endTime = chrono::high_resolution_clock::now();

        licensePlate->setRecognizerTime((double) chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count());
        auto stop = 1;
        auto [licensePlateLabel, probability] = getLicensePlateLabel(recognizerResult, licensePlate->isSquare());

        bool isValid = isValidLicensePlate(licensePlateLabel, probability);
        cameraToOverallPlatesCount.at(licensePlate->getCameraIp()) += 1;

        if (!isValid) {
            bool isValidByConfidence = isValidByConf(probability);
            bool isValidByLPPattern = isValidByPattern(licensePlateLabel);

            if (!isValidByConfidence && isValidByLPPattern) {
                cameraToNotValidByConfPlatesCount.at(licensePlate->getCameraIp()) += 1;
                cameraToNotValidProbs.at(licensePlate->getCameraIp()).push_back(probability);
            } else if (isValidByConfidence && !isValidByLPPattern) {
                cameraToNotValidByPatternPlatesCount.at(licensePlate->getCameraIp()) += 1;
            } else {
                cameraToNotValidBothPlatesCount.at(licensePlate->getCameraIp()) += 1;
                cameraToNotValidProbs.at(licensePlate->getCameraIp()).push_back(probability);
            }
            continue;
        };

        licensePlate->setPlateProbability(probability);

        cameraToValidProbs.at(licensePlate->getCameraIp()).push_back(probability);

        for_each(licensePlateLabel.begin(), licensePlateLabel.end(), Utils::convert());


        if (DEBUG && IMSHOW) {
            cv::Mat displayImage;
            cv::resize(licensePlate->getPlateImage(), displayImage,
                       cv::Size(licensePlate->getPlateImage().cols * 3.5, licensePlate->getPlateImage().rows * 3.5), 0,
                       0);
            cv::imshow(licensePlate->getCameraIp() + "_licensePlate", displayImage);
            cv::waitKey(60);
        }


        cameraIpToPlateWhitenessSum.at(licensePlate->getCameraIp()) += (int) licensePlate->getWhiteness();
        cameraIpToPlateDFTSharpnessSum.at(licensePlate->getCameraIp()) += (int) licensePlate->getDFTSharpness();
        cameraIpToPlateLaplacianSharpnessSum.at(licensePlate->getCameraIp()) += (int) licensePlate->getSharpness();
        cameraIpToCurrentCameraWhiteness.at(licensePlate->getCameraIp()) = (int) cv::mean(
                licensePlate->getCarImage())[0];

        cameraIpToPlateWhitenessNum.at(licensePlate->getCameraIp()) += 1;


        if (licensePlate->getWhiteness() > Constants::LP_WHITENESS_MAX)
            cameraIpToWhitePlatesNum.at(licensePlate->getCameraIp()) += 1;
        else if (licensePlate->getWhiteness() < Constants::LP_WHITENESS_MIN)
            cameraIpToDarkPlatesNum.at(licensePlate->getCameraIp()) += 1;

        cameraToPassPlatesCount.at(licensePlate->getCameraIp()) += 1;

        licensePlate->setLicensePlateLabel(std::move(licensePlateLabel));


        addPlateToTrack(licensePlate);


    }
}

CameraScope LPRecognizerService::getCameraScope(const string &cameraIp) const {
    int index = 0;
    for (int i = 0; i < cameraScopeVector.size(); ++i) {
        auto item = cameraScopeVector[i];
        if (item.getCameraIp() == cameraIp)
            index = i;
    }
    return cameraScopeVector[index];

}

void LPRecognizerService::shutdown() {
    LOG_INFO("service shutting down");
    shutdownFlag = true;
    lpQueue->push(nullptr);
    shutdownEvent.notify_one();
    if (backgroundThread.joinable())
        backgroundThread.join();
    for (auto &cameraIpToCarTracker: cameraIpToCarTrackerMap) {
        cameraIpToCarTracker.second->shutdown();
    }
}

bool LPRecognizerService::isChooseThisFrame() {
    srand(time(nullptr));
    auto randomNumber = 1 + rand() % 100; // generating number between 1 and 100
    return (randomNumber < 51);
}

void LPRecognizerService::saveFrame(const shared_ptr<LicensePlate> &plate) {
    if (!isChooseThisFrame()) return;

    string fileName = RandomStringGenerator::generate(30, Constants::IMAGE_DIRECTORY, Constants::JPG_EXTENSION);
    auto frame = plate->getCarImage();
    cv::imwrite(fileName, frame);
}

void LPRecognizerService::checkCurrentPlatesStatus() {
    unique_lock<mutex> lock(checkCurrentAuto);
    auto timeout = chrono::minutes(waitMinutes);
    while (!shutdownFlag) {
        if (!shutdownEvent.wait_for(lock, timeout, [this] { return shutdownFlag; })) {
            for (auto &it: cameraScopeVector) {
                string ipCamera = it.getCameraIp();

                int overallCount = cameraToOverallPlatesCount.at(ipCamera);
                int passedCount = cameraToPassPlatesCount.at(ipCamera);
                int notValidByConfidence = cameraToNotValidByConfPlatesCount.at(ipCamera);
                int notValidByPattern = cameraToNotValidByPatternPlatesCount.at(ipCamera);
                int notValidByBoth = cameraToNotValidBothPlatesCount.at(ipCamera);

                LOG_INFO(
                        "Statistics 1 - %s: overall: %d, passed: %d, not_val_confidence: %d, not_val_pattern: %d, not_val_both: %d",
                        ipCamera.data(), overallCount,
                        passedCount, notValidByConfidence, notValidByPattern, notValidByBoth);

                for (const auto &camera: cameraScopeVector)
                    if (cameraIpToPlateWhitenessNum.at(camera.getCameraIp()) > 0) {
                        const string &cameraIp = camera.getCameraIp();

                        auto probs = cameraToValidProbs.at(cameraIp);
                        auto notValidProbs = cameraToNotValidProbs.at(cameraIp);

                        double meanValValid = probs.empty() ? 0 : mean(probs);
                        double mediaValValid = probs.empty() ? 0 : median(probs);

                        double meanValNotValid = notValidProbs.empty() ? 0 : mean(notValidProbs);
                        double mediaValNotValid = notValidProbs.empty() ? 0 : median(notValidProbs);


                        LOG_INFO(
                                "Statistics 2 - %s: frameWhiteness: %0.1f, plateMeanWhiteness: %.2f, plateLaplacianSharpness: %.2f",
                                cameraIp.data(),
                                cameraIpToCurrentCameraWhiteness.at(cameraIp),
                                cameraIpToPlateWhitenessSum.at(cameraIp) / cameraIpToPlateWhitenessNum.at(cameraIp),
                                cameraIpToPlateLaplacianSharpnessSum.at(cameraIp) /
                                cameraIpToPlateWhitenessNum.at(cameraIp));

                        LOG_INFO(
                                "Statistics 3 - %s: platesWhitenessCount: %0.1f, whitePlates: %0.1f, darkPlates: %0.1f",
                                cameraIp.data(), cameraIpToPlateWhitenessNum.at(cameraIp),
                                cameraIpToWhitePlatesNum.at(cameraIp), cameraIpToDarkPlatesNum.at(cameraIp));

                        LOG_INFO(
                                "Statistics 4 - %s: mean_val_prob: %0.1f, median_val_prob: %0.1f, mean_not_val_prob: %0.1f, median_not_val_prob: %0.1f",
                                cameraIp.data(), meanValValid, mediaValValid, meanValNotValid, mediaValNotValid);
                        cameraIpToCurrentCameraWhiteness.at(cameraIp) = 0;
                        cameraIpToPlateWhitenessSum.at(cameraIp) = 0;
                        cameraIpToPlateWhitenessNum.at(cameraIp) = 0;
                        cameraIpToPlateDFTSharpnessSum.at(cameraIp) = 0;
                        cameraIpToPlateLaplacianSharpnessSum.at(cameraIp) = 0;
                        cameraIpToWhitePlatesNum.at(cameraIp) = 0;
                        cameraIpToDarkPlatesNum.at(cameraIp) = 0;
                        cameraToValidProbs.at(cameraIp) = {};
                        cameraToNotValidProbs.at(cameraIp) = {};
                    }
                cameraToOverallPlatesCount.at(ipCamera) = 0;
                cameraToPassPlatesCount.at(ipCamera) = 0;
                cameraToNotValidByConfPlatesCount.at(ipCamera) = 0;
                cameraToNotValidByPatternPlatesCount.at(ipCamera) = 0;
                cameraToNotValidBothPlatesCount.at(ipCamera) = 0;
            }
        }
    }
}

bool LPRecognizerService::isValidByPattern(const string &lpLabel) {
    auto plateCountry = templateMatching->getCountryCode(lpLabel);
    auto isTemplateMatched = plateCountry != Constants::UNIDENTIFIED_COUNTRY;
    return isTemplateMatched;
}

bool LPRecognizerService::isValidByConf(float probability) const {
    return probability > RECOGNIZER_THRESHOLD;
}

double LPRecognizerService::mean(vector<double> &nums) {
    double sum = 0.0;
    for (double value: nums) {
        sum += value;
    }
    double mean = sum / nums.size();

    return mean * 100;
}

double LPRecognizerService::median(vector<double> &nums) {
    std::vector<int> integerValues;
    std::transform(nums.begin(), nums.end(), std::back_inserter(integerValues),
                   [](double prob) { return static_cast<int>(prob * 100); });

    std::sort(integerValues.begin(), integerValues.end());
    double median;
    if (integerValues.size() % 2 == 0) {
        median = (integerValues[integerValues.size() / 2 - 1] + integerValues[integerValues.size() / 2]) / 2.0;
    } else {
        median = integerValues[integerValues.size() / 2];
    }

    return median;
}

cv::Mat LPRecognizerService::combineInOneLineSquareLpPlate(const cv::Mat &lpImage) {
    auto blackImage = cv::Mat(Constants::RECT_LP_H, Constants::BLACK_IMG_WIDTH, CV_8UC3,
                              cv::Scalar(0, 0, 0));
    auto topHalf = lpImage(cv::Rect(0, 0, Constants::SQUARE_LP_W, Constants::SQUARE_LP_H / 2));
    auto bottomHalf = lpImage(cv::Rect(0, Constants::SQUARE_LP_H / 2, Constants::SQUARE_LP_W,
                                       Constants::SQUARE_LP_H / 2));

    cv::Mat combinedPlateImage; // combining top and bottom half of square lp in one line
    cv::hconcat(topHalf, blackImage, topHalf);
    cv::hconcat(topHalf, bottomHalf, combinedPlateImage);
    return combinedPlateImage;
}
