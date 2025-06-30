import React from 'react';

import { useI18n } from '@milesight/shared/src/hooks';

import { Empty } from '@/components';

export interface ChartEmptyWrapperProps {
    isEmptyData?: boolean;
    children: React.ReactNode;
}

/**
 * whether to render chart empty ui
 */
const ChartEmptyWrapper: React.FC<ChartEmptyWrapperProps> = props => {
    const { isEmptyData, children } = props;

    const { getIntlText } = useI18n();

    if (isEmptyData) {
        return (
            <div className="chart-item__empty">
                <Empty size="small" text={getIntlText('common.label.empty')} />
            </div>
        );
    }

    return children;
};

export default ChartEmptyWrapper;
