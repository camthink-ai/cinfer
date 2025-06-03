import React, { useRef, useEffect } from 'react';

import { useI18n } from '@milesight/shared/src/hooks';

import { EchartsUI } from '@/components';
import { useEcharts } from '@/components/echarts';

/**
 * model status chart
 */
const ModelStatus: React.FC = () => {
    const { getIntlText } = useI18n();

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
                        { name: getIntlText('common.label.already_publish_status'), value: 1048 },
                        {
                            name: getIntlText('common.label.unpublish_status'),
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
                    name: getIntlText('dashboard.label.model_operating_status'),
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
