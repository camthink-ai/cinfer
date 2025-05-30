import React from 'react';

import { useI18n } from '@milesight/shared/src/hooks';

import { BasicInfo, CPUUsage, ModelStatus } from './components';

import './style.less';

const Dashboard: React.FC = () => {
    const { getIntlText } = useI18n();

    return (
        <div className="ms-main-blank">
            <div className="ms-view-dashboard">
                <div className="dashboard-basic-info">
                    <BasicInfo />
                </div>
                <div className="dashboard-charts">
                    <div className="chart-container">
                        <div className="chart-item">
                            <div className="chart-item__title">
                                {getIntlText('dashboard.label.cpu_usage_title')}
                            </div>
                            <CPUUsage />
                        </div>
                        <div className="chart-item">2</div>
                    </div>
                    <div className="chart-container">
                        <div className="chart-item">3</div>
                        <div className="chart-item">
                            <div className="chart-item__model-title">
                                {getIntlText('dashboard.label.model_operating_status')}
                            </div>
                            <div className="chart-item__model-wrapper">
                                <ModelStatus />
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Dashboard;
