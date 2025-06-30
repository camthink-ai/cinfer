import React from 'react';

import { Logo, TopBarMenus, type TopBarMenusProps, MoreUserInfo } from '@/components';

export interface LayoutHeaderProps {
    menus?: TopBarMenusProps[];
}

/**
 * the layout header
 */
const LayoutHeader: React.FC<LayoutHeaderProps> = props => {
    const { menus } = props;

    return (
        <div className="ms-layout-header">
            <div className="ms-layout-header__left">
                <Logo />
                <TopBarMenus menus={menus} />
            </div>

            <MoreUserInfo />
        </div>
    );
};

export default LayoutHeader;
