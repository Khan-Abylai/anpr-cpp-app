//
// Created by Камалхан Артыкбаев on 14.10.2022.
//

#include "CameraScope.h"

CameraScope::CameraScope(const std::string &cameraIp, const std::string &rtspUrl, const std::string &resultSentIp,
                         const std::string &nodeIp, const std::string &snapshotSentIp,
                         const std::string &secondaryResultSentEndpoint, bool useSecondaryEndpoint,
                         const std::string &secondarySnapshotSentEndpoint, bool useSecondarySnapshotEndpoint,
                         Constants::VehicleClassificationType typeOfCarRecognition, float timeBetweenResendingPlates,
                         int platesCount,
                         bool useMask2,
                         bool useDirection, const std::string &originPoint, bool isForward,
                         DimensionCoords dimensionCoords) {
    this->cameraIp = cameraIp;
    this->rtspUrl = rtspUrl;
    this->resultSendIp = resultSentIp;
    this->nodeIp = nodeIp;
    this->snapshotSendIp = snapshotSentIp;
    this->otherResultSendEndpoint = secondaryResultSentEndpoint;
    this->otherSnapshotSendEndpoint = secondarySnapshotSentEndpoint;
    this->useSecondaryEndpoint = useSecondaryEndpoint;
    this->useSecondarySnapshotEndpoint = useSecondarySnapshotEndpoint;
    this->vehicleClassificationType = typeOfCarRecognition;
    this->timeBetweenResendingPlates = timeBetweenResendingPlates;
    this->platesCount = platesCount;
    this->useMask2Flag = useMask2;
    this->useDirectionFlag = useDirection;
    this->originPoint = originPoint;
    this->isForwardFlag = isForward;
    this->COORDS = dimensionCoords;
}

const std::string &CameraScope::getCameraIp() const {
    return cameraIp;
}

const std::string &CameraScope::getRtspUrl() const {
    return rtspUrl;
}

const std::string &CameraScope::getResultSendIp() const {
    return resultSendIp;
}

const std::string &CameraScope::getNodeIp() const {
    return nodeIp;
}

const std::string &CameraScope::getSnapshotSendIp() const {
    return snapshotSendIp;
}

const std::string &CameraScope::getOtherResultSendEndpoint() const {
    return otherResultSendEndpoint;
}

const std::string &CameraScope::getOtherSnapshotSendEndpoint() const {
    return otherSnapshotSendEndpoint;
}

const std::string &CameraScope::getOriginPoint() const {
    return originPoint;
}

bool CameraScope::isUseSecondaryEndpoint() const {
    return useSecondaryEndpoint;
}

bool CameraScope::isUseSecondarySnapshotEndpoint() const {
    return useSecondarySnapshotEndpoint;
}


bool CameraScope::isUseMask2Flag() const {
    return useMask2Flag;
}

bool CameraScope::isUseDirectionFlag() const {
    return useDirectionFlag;
}

bool CameraScope::isForwardFlagEnabled() const {
    return isForwardFlag;
}

float CameraScope::getTimeBetweenResendingPlates() const {
    return timeBetweenResendingPlates;
}

int CameraScope::getPlatesCount() const {
    return platesCount;
}

DimensionCoords CameraScope::getDimensionCoords() const {
    return COORDS;
}

Constants::VehicleClassificationType CameraScope::whatTypeOfRecognitionEnabled() const {
    return vehicleClassificationType;
}
