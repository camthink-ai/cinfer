import React, { useRef, useEffect } from 'react';

import { EchartsUI } from '@/components';
import { useEcharts } from '@/components/echarts';

/**
 * model status chart
 */
const ModelStatus: React.FC = () => {
    const modelChartRef = useRef<HTMLDivElement>(null);

    const { renderEcharts } = useEcharts(modelChartRef);

    useEffect(() => {
        renderEcharts({
            legend: {
                bottom: '2%',
                left: 'center',
            },
            series: [
                {
                    animationDelay() {
                        return Math.random() * 100;
                    },
                    animationEasing: 'exponentialInOut',
                    animationType: 'scale',
                    avoidLabelOverlap: false,
                    color: ['#FF7C42', '#F2F3F5'],
                    data: [
                        { name: '已发布', value: 1048 },
                        {
                            name: '未发布',
                            value: 735,
                            emphasis: { itemStyle: { color: '#F2F3F5' } },
                        },
                    ],
                    emphasis: {
                        label: {
                            fontSize: '12',
                            fontWeight: 'bold',
                            show: true,
                        },
                    },
                    label: {
                        position: 'center',
                        show: false,
                    },
                    labelLine: {
                        show: false,
                    },
                    name: '模型运行状态',
                    radius: ['40%', '65%'],
                    type: 'pie',
                },
            ],
            tooltip: {
                trigger: 'item',
            },
        });
    }, []);

    return <EchartsUI ref={modelChartRef} height="280px" />;
};

export default ModelStatus;
