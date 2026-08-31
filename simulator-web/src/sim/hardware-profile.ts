export interface HardwareSensorProfile {
  id: string;
  cameraHz: number;
  lidarMaxRangeM: number;
  lidarRangeNoiseStdM: number;
  lidarRandomDropoutProbability: number;
  lidarDropoutVariationStd: number;
  lidarBlindSectorMinDeg: number;
  lidarBlindSectorMaxDeg: number;
  odometryLinearScale: number;
  odometryAxisSkewDeg: number;
  odometryAngularScale: number;
  odometryVelocityNoiseStdMps: number;
  imuYawScale: number;
  imuYawDriftDegPerSecond: number;
  imuYawRandomWalkDegPerSqrtSecond: number;
  imuYawNoiseStdDeg: number;
}

/**
 * Conservative hardware profile derived from the retained Kiwi map bundles.
 *
 * The saved robot maps contain only accepted, filtered scans, so the values
 * intentionally reproduce their observable envelope rather than claiming to
 * model packet-level failure modes that were not recorded.
 */
export const RETAINED_ROBOT_PROFILE: Readonly<HardwareSensorProfile> =
  Object.freeze({
    id: "retained-robot-maps-v1",
    cameraHz: 9.69,
    lidarMaxRangeM: 8,
    lidarRangeNoiseStdM: 0.003,
    // The home geometry already produces most of the no-returns seen in the
    // saved scans. This is the residual loss needed on top of that geometry.
    lidarRandomDropoutProbability: 0.06,
    lidarDropoutVariationStd: 0.02,
    lidarBlindSectorMinDeg: 5,
    lidarBlindSectorMaxDeg: 12,
    odometryLinearScale: 1.04,
    odometryAxisSkewDeg: 0.35,
    odometryAngularScale: 1.005,
    odometryVelocityNoiseStdMps: 0.004,
    imuYawScale: 1.003,
    imuYawDriftDegPerSecond: 0.03,
    imuYawRandomWalkDegPerSqrtSecond: 0.015,
    imuYawNoiseStdDeg: 0.12,
  });

export const IDEAL_SENSOR_PROFILE: Readonly<HardwareSensorProfile> =
  Object.freeze({
    id: "ideal",
    cameraHz: 10,
    lidarMaxRangeM: 12,
    lidarRangeNoiseStdM: 0,
    lidarRandomDropoutProbability: 0,
    lidarDropoutVariationStd: 0,
    lidarBlindSectorMinDeg: 0,
    lidarBlindSectorMaxDeg: 0,
    odometryLinearScale: 1,
    odometryAxisSkewDeg: 0,
    odometryAngularScale: 1,
    odometryVelocityNoiseStdMps: 0,
    imuYawScale: 1,
    imuYawDriftDegPerSecond: 0,
    imuYawRandomWalkDegPerSqrtSecond: 0,
    imuYawNoiseStdDeg: 0,
  });
