import React, { useRef, useEffect } from 'react';

import { EchartsUI } from '@/components';
import { useEcharts } from '@/components/echarts';

export interface MemoryUsageProps {
    memoryData: (string | number)[][];
}

/**
 * memory usage chart
 */
const MemoryUsage: React.FC<MemoryUsageProps> = props => {
    const { memoryData } = props;

    const memoryChartRef = useRef<HTMLDivElement>(null);

    const { renderEcharts } = useEcharts(memoryChartRef);

    useEffect(() => {
        renderEcharts({
            grid: {
                containLabel: true,
                top: '42px',
                left: '1%',
            },
            title: {
                text: `${memoryData?.[memoryData.length - 1]?.[1] || 0}%`,
                textAlign: 'left',
            },
            series: [
                {
                    type: 'line',
                    smooth: true,
                    itemStyle: {
                        color: '#1EBA62',
                    },
                    data: memoryData,
                },
            ],
            tooltip: {
                axisPointer: {
                    lineStyle: {
                        color: '#1EBA62',
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
                    fillerColor: 'rgba(30, 186, 98, 0.15)',
                    showDetail: false,
                    moveHandleStyle: {
                        color: '#1EBA62',
                        opacity: 0.16,
                    },
                    emphasis: {
                        handleLabel: {
                            show: true,
                        },
                        moveHandleStyle: {
                            color: '#1EBA62',
                            opacity: 1,
                        },
                    },
                    borderColor: '#E5E6EB',
                    dataBackground: {
                        lineStyle: {
                            color: '#1EBA62',
                            opacity: 0.36,
                        },
                        areaStyle: {
                            color: '#1EBA62',
                            opacity: 0.08,
                        },
                    },
                    selectedDataBackground: {
                        lineStyle: {
                            color: '#1EBA62',
                            opacity: 0.8,
                        },
                        areaStyle: {
                            color: '#1EBA62',
                            opacity: 0.2,
                        },
                    },
                    brushStyle: {
                        color: '#1EBA62',
                        opacity: 0.16,
                    },
                },
            ],
        });
    }, [memoryData]);

    return <EchartsUI ref={memoryChartRef} height="280px" />;
};

export default MemoryUsage;
