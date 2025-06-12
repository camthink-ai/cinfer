import React, { useRef, useEffect } from 'react';

import { EchartsUI } from '@/components';
import { useEcharts } from '@/components/echarts';

export interface GPUUsageProps {
    gpuData: (string | number)[][];
}

/**
 * gpu usage chart
 */
const GPUUsage: React.FC<GPUUsageProps> = props => {
    const { gpuData } = props;

    const gpuChartRef = useRef<HTMLDivElement>(null);

    const { renderEcharts } = useEcharts(gpuChartRef);

    useEffect(() => {
        renderEcharts({
            grid: {
                containLabel: true,
                top: '42px',
                left: '1%',
            },
            title: {
                text: `${gpuData?.[gpuData.length - 1]?.[1] || 0}%`,
                textAlign: 'left',
            },
            series: [
                {
                    type: 'line',
                    smooth: true,
                    itemStyle: {
                        color: '#3491FA',
                    },
                    data: gpuData,
                },
            ],
            tooltip: {
                axisPointer: {
                    lineStyle: {
                        color: '#3491FA',
                        width: 1,
                    },
                },
                trigger: 'axis',
            },
            xAxis: {
                type: 'time',
                boundaryGap: [0, 0],
            },
            yAxis: {
                axisTick: {
                    show: false,
                },
                max: 100,
                interval: 25,
                type: 'value',
                boundaryGap: [0, '100%'],
                axisLabel: {
                    formatter(value) {
                        return `${value}%`;
                    },
                },
            },
            dataZoom: [
                {
                    type: 'inside',
                    start: 0,
                    end: 100,
                },
                {
                    type: 'slider',
                    start: 0,
                    end: 100,
                    fillerColor: 'rgba(52, 145, 250, 0.15)',
                    showDetail: false,
                    moveHandleStyle: {
                        color: '#3491FA',
                        opacity: 0.16,
                    },
                    emphasis: {
                        handleLabel: {
                            show: true,
                        },
                        moveHandleStyle: {
                            color: '#3491FA',
                            opacity: 1,
                        },
                    },
                    borderColor: '#E5E6EB',
                    dataBackground: {
                        lineStyle: {
                            color: '#3491FA',
                            opacity: 0.36,
                        },
                        areaStyle: {
                            color: '#3491FA',
                            opacity: 0.08,
                        },
                    },
                    selectedDataBackground: {
                        lineStyle: {
                            color: '#3491FA',
                            opacity: 0.8,
                        },
                        areaStyle: {
                            color: '#3491FA',
                            opacity: 0.2,
                        },
                    },
                    brushStyle: {
                        color: '#3491FA',
                        opacity: 0.16,
                    },
                },
            ],
        });
    }, [gpuData]);

    return <EchartsUI ref={gpuChartRef} height="280px" />;
};

export default GPUUsage;
