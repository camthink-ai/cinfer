import React, { useMemo, useState } from 'react';
import { useMemoizedFn } from 'ahooks';
import { merge } from 'lodash-es';

import { TextField, type TextFieldProps, InputAdornment, IconButton } from '@mui/material';
import { VisibilityIcon, VisibilityOffIcon, HttpsIcon } from '@milesight/shared/src/components';

export type PasswordInputProps = TextFieldProps & {
    /**
     * To show the default prefix icon
     */
    showDefaultPrefixIcon?: boolean;
};

/**
 * Password Input Components
 */
const PasswordInput: React.FC<PasswordInputProps> = props => {
    const { showDefaultPrefixIcon = false, slotProps, ...restProps } = props;
    const [showPassword, setShowPassword] = useState(false);

    const handleClickShowPassword = useMemoizedFn(() => setShowPassword(show => !show));

    const handleMouseDownPassword = useMemoizedFn((event: React.MouseEvent<HTMLButtonElement>) => {
        event.preventDefault();
    });

    const handleMouseUpPassword = useMemoizedFn((event: React.MouseEvent<HTMLButtonElement>) => {
        event.preventDefault();
    });

    /**
     * password input the prefix icon
     */
    const prefixIcon = useMemo(() => {
        if (!showDefaultPrefixIcon) return {};

        return {
            startAdornment: (
                <InputAdornment position="start">
                    <HttpsIcon />
                </InputAdornment>
            ),
        };
    }, [showDefaultPrefixIcon]);

    return (
        <TextField
            {...restProps}
            type={showPassword ? 'text' : 'password'}
            slotProps={merge({}, slotProps, {
                input: {
                    ...prefixIcon,
                    endAdornment: (
                        <InputAdornment position="end">
                            <IconButton
                                onClick={handleClickShowPassword}
                                onMouseDown={handleMouseDownPassword}
                                onMouseUp={handleMouseUpPassword}
                                edge="end"
                            >
                                {showPassword ? <VisibilityIcon /> : <VisibilityOffIcon />}
                            </IconButton>
                        </InputAdornment>
                    ),
                },
            })}
        />
    );
};

export default PasswordInput;
