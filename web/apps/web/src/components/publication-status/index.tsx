import React, { useMemo } from 'react';
import classNames from 'classnames';

import { useI18n } from '@milesight/shared/src/hooks';

import './style.less';

export interface PublicationStatusProps {
    hasPublished?: boolean;
}

const PublicationStatus: React.FC<PublicationStatusProps> = props => {
    const { hasPublished } = props;
    const { getIntlText } = useI18n();

    const publicationStatusCls = useMemo(() => {
        return classNames('publication-status', {
            published: hasPublished,
        });
    }, [hasPublished]);

    return (
        <div className={publicationStatusCls}>
            <div className="publication-status__circle" />
            <div className="publication-status__text">
                {hasPublished
                    ? getIntlText('common.label.already_publish_status')
                    : getIntlText('common.label.unpublish_status')}
            </div>
        </div>
    );
};

export default PublicationStatus;
