import React from 'react';
import { useMatches } from 'react-router';
import { Link } from 'react-router-dom';
import { MenuList, MenuItem } from '@mui/material';

import './style.less';

export interface TopBarMenusProps {
    name: string;
    path: string;
    icon?: React.ReactNode;
}

export interface Props {
    menus?: TopBarMenusProps[];
}

/**
 * top bar menus
 */
const TopBarMenus: React.FC<Props> = props => {
    const { menus } = props;

    const routes = useMatches().slice(1);
    const selectedKeys = routes.map(route => route.pathname);

    return (
        <div className="ms-top-bar-menus">
            <MenuList className="ms-top-bar-menus__list">
                {menus?.map(menu => (
                    <MenuItem
                        disableRipple
                        key={menu.path}
                        className="ms-top-bar-menus__item"
                        selected={selectedKeys.includes(menu.path)}
                        sx={{
                            padding: 0,
                        }}
                    >
                        <Link className="ms-top-bar-menus__item-link" to={menu.path}>
                            {menu.icon}
                            <span className="ms-name">{menu.name}</span>
                        </Link>
                    </MenuItem>
                ))}
            </MenuList>
        </div>
    );
};

export default TopBarMenus;
