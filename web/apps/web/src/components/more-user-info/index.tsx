import React from 'react';
import PopupState, { bindTrigger, bindMenu } from 'material-ui-popup-state';
import {
    Menu,
    MenuItem,
    Avatar,
    ListItem,
    ListItemAvatar,
    ListItemText,
    ListItemIcon,
    Divider,
    Stack,
} from '@mui/material';
import { useNavigate } from 'react-router-dom';
import { merge } from 'lodash-es';
import { useMemoizedFn } from 'ahooks';

import { LogoutIcon } from '@milesight/shared/src/components';
import { iotLocalStorage, TOKEN_CACHE_KEY } from '@milesight/shared/src/utils/storage';
import { useI18n } from '@milesight/shared/src/hooks';
import { type TokenDataType } from '@milesight/shared/src/utils/request';

import Tooltip from '@/components/tooltip';
import { useUserStore } from '@/stores';
import { globalAPI, awaitWrap, isRequestSuccess } from '@/services/http';
import LangItem from './lang-item';

import './style.less';

function stringToColor(string: string) {
    let hash = 0;
    let i;

    /* eslint-disable no-bitwise */
    for (i = 0; i < string.length; i += 1) {
        hash = string.charCodeAt(i) + ((hash << 5) - hash);
    }

    let color = '#';

    for (i = 0; i < 3; i += 1) {
        const value = (hash >> (i * 8)) & 0xff;
        color += `00${value.toString(16)}`.slice(-2);
    }
    /* eslint-enable no-bitwise */

    return color;
}

function stringAvatar(name: string) {
    return {
        sx: {
            width: 32,
            height: 32,
            bgcolor: stringToColor(name),
        },
        children: `${name.split(' ')[0][0]}`,
    };
}

/**
 * User information display and operation components
 */
const MoreUserInfo: React.FC = () => {
    const navigate = useNavigate();
    const { getIntlText } = useI18n();
    const { setUserInfo, userInfo } = useUserStore();

    /**
     * user logout
     */
    const handleUserLogout = useMemoizedFn(async () => {
        const token = iotLocalStorage.getItem<TokenDataType>(TOKEN_CACHE_KEY);
        if (!token?.refresh_token) return;

        const [error, resp] = await awaitWrap(
            globalAPI.userLogout({
                refresh_token: token.refresh_token,
            }),
        );
        if (error || !isRequestSuccess(resp)) return;

        setUserInfo(null);
        iotLocalStorage.removeItem(TOKEN_CACHE_KEY);
        navigate('/auth/login');
    });

    return (
        <PopupState variant="popover" popupId="user-info-menu">
            {state => (
                <div className="ms-user-info">
                    <Stack
                        direction="row"
                        spacing={2}
                        alignItems="center"
                        className="ms-user-info__trigger"
                        {...bindTrigger(state)}
                    >
                        <Avatar {...stringAvatar(userInfo?.username || '')} />
                    </Stack>
                    <Menu
                        {...bindMenu(state)}
                        anchorOrigin={{
                            vertical: 'bottom',
                            horizontal: 'right',
                        }}
                        transformOrigin={{
                            vertical: 'top',
                            horizontal: 'right',
                        }}
                    >
                        <ListItem sx={{ width: 255 }} alignItems="center">
                            <ListItemAvatar className="ms-user-info__avatar">
                                <Avatar
                                    {...merge(stringAvatar(userInfo?.username || ''), {
                                        sx: { width: 44, height: 44 },
                                    })}
                                />
                            </ListItemAvatar>
                            <ListItemText
                                className="ms-user-info__text"
                                primary={<Tooltip title={userInfo?.username || ''} autoEllipsis />}
                            />
                        </ListItem>
                        <Divider sx={{ marginBottom: '8px' }} className="ms-user-info__divider" />
                        <LangItem onChange={() => state.close()} />
                        <MenuItem
                            onClick={() => {
                                state.close();
                                /** Sign out logic */
                                handleUserLogout();
                            }}
                        >
                            <ListItemIcon>
                                <LogoutIcon />
                            </ListItemIcon>
                            {getIntlText('common.label.sign_out')}
                        </MenuItem>
                    </Menu>
                </div>
            )}
        </PopupState>
    );
};

export default MoreUserInfo;
