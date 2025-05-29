import React from 'react';
import { BasicInfo, CPUUsage, ModelStatus } from './components';

import './style.less';

const Dashboard: React.FC = () => {
    return (
        <div className="ms-main-blank">
            <div className="ms-view-dashboard">
                <div className="dashboard-basic-info">
                    <BasicInfo />
                </div>
                <div className="dashboard-charts">
                    <div className="chart-container">
                        <div className="chart-item">
                            <div className="chart-item__title">CPU 使用率</div>
                            <CPUUsage />
                        </div>
                        <div className="chart-item">2</div>
                    </div>
                    <div className="chart-container">
                        <div className="chart-item">3</div>
                        <div className="chart-item">
                            <div className="chart-item__title">模型运行状态</div>
                            <ModelStatus />
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Dashboard;
