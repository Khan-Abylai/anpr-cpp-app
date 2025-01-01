//
// Created by Камалхан Артыкбаев on 14.10.2022.
//
#pragma once

#include <regex>
#include <string>
#include "Constants.h"

struct DimensionCoords {
    float x1;
    float y1;

    float x2;
    float y2;
};

class CameraScope {
public:
    CameraScope(const std::string &cameraIp, const std::string &rtspUrl, const std::string &resultSentIp,
                const std::string &nodeIp, const std::string &snapshotSentIp,
                const std::string &secondaryResultSentEndpoint, bool useSecondaryEndpoint,
                const std::string &secondarySnapshotSentEndpoint, bool useSecondarySnapshotEndpoint,
                Constants::VehicleClassificationType typeOfCarRecognition,
                float timeBetweenResendingPlates, int platesCount, bool useMask2, bool useDirection,
                const std::string &originPoint, bool isForward,
                DimensionCoords dimensionCoords);


private:
    std::string cameraIp, rtspUrl, resultSendIp, nodeIp, snapshotSendIp, otherResultSendEndpoint, otherSnapshotSendEndpoint, originPoint;
    bool useSecondaryEndpoint, useSecondarySnapshotEndpoint, useMask2Flag, useDirectionFlag, isForwardFlag;
    float timeBetweenResendingPlates;
    int platesCount;
    DimensionCoords COORDS{};
    Constants::VehicleClassificationType vehicleClassificationType;
public:
    [[nodiscard]] const std::string &getCameraIp() const;

    [[nodiscard]] const std::string &getRtspUrl() const;

    [[nodiscard]] const std::string &getResultSendIp() const;

    [[nodiscard]] const std::string &getNodeIp() const;

    [[nodiscard]] const std::string &getSnapshotSendIp() const;

    [[nodiscard]] const std::string &getOtherResultSendEndpoint() const;

    [[nodiscard]] const std::string &getOtherSnapshotSendEndpoint() const;

    [[nodiscard]] const std::string &getOriginPoint() const;

    [[nodiscard]] bool isUseSecondaryEndpoint() const;

    [[nodiscard]] bool isUseSecondarySnapshotEndpoint() const;

    [[nodiscard]] Constants::VehicleClassificationType whatTypeOfRecognitionEnabled() const;

    [[nodiscard]] bool isUseMask2Flag() const;

    [[nodiscard]] bool isUseDirectionFlag() const;

    [[nodiscard]] bool isForwardFlagEnabled() const;

    [[nodiscard]] float getTimeBetweenResendingPlates() const;

    [[nodiscard]] int getPlatesCount() const;

    [[nodiscard]] DimensionCoords getDimensionCoords() const;

};