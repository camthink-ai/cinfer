import React from 'react';
import { isEmpty } from 'lodash-es';

import { useI18n } from '@milesight/shared/src/hooks';

import {
    BasicInfo,
    CPUUsage,
    ModelStatus,
    GPUUsage,
    MemoryUsage,
    ChartEmptyWrapper,
} from './components';
import { useSystemInfo, useSystemMetrics } from './hooks';

import './style.less';

const Dashboard: React.FC = () => {
    const { getIntlText } = useI18n();
    const { systemInfo, modelsStatus } = useSystemInfo();
    const { cpuData, gpuData, memoryData } = useSystemMetrics();

    return (
        <div className="ms-main-blank">
            <div className="ms-view-dashboard">
                <div className="dashboard-basic-info">
                    <BasicInfo data={systemInfo} />
                </div>
                <div className="dashboard-charts">
                    <div className="chart-container">
                        <div className="chart-item">
                            <div className="chart-item__title">
                                {getIntlText('dashboard.label.cpu_usage_title')}
                            </div>
                            <ChartEmptyWrapper
                                isEmptyData={!Array.isArray(cpuData) || isEmpty(cpuData)}
                            >
                                <CPUUsage cpuData={cpuData} />
                            </ChartEmptyWrapper>
                        </div>
                        <div className="chart-item">
                            <div className="chart-item__title">
                                {getIntlText('dashboard.label.gpu_usage_title')}
                            </div>
                            <ChartEmptyWrapper
                                isEmptyData={!Array.isArray(gpuData) || isEmpty(gpuData)}
                            >
                                <GPUUsage gpuData={gpuData} />
                            </ChartEmptyWrapper>
                        </div>
                    </div>
                    <div className="chart-container">
                        <div className="chart-item">
                            <div className="chart-item__title">
                                {getIntlText('dashboard.label.memory_usage_title')}
                            </div>
                            <ChartEmptyWrapper
                                isEmptyData={!Array.isArray(memoryData) || isEmpty(memoryData)}
                            >
                                <MemoryUsage memoryData={memoryData} />
                            </ChartEmptyWrapper>
                        </div>
                        <div className="chart-item">
                            <div className="chart-item__model-title">
                                {getIntlText('dashboard.label.model_operating_status')}
                            </div>
                            <div className="chart-item__model-wrapper">
                                <ChartEmptyWrapper
                                    isEmptyData={
                                        !modelsStatus?.published_count &&
                                        !modelsStatus?.unpublished_count
                                    }
                                >
                                    <ModelStatus status={modelsStatus} />
                                </ChartEmptyWrapper>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Dashboard;
