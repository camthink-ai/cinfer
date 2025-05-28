import { useState, useMemo, useCallback } from 'react';
import { type ControllerProps } from 'react-hook-form';
import { TextField, IconButton, InputAdornment, type TextFieldProps } from '@mui/material';
import { useI18n } from '@milesight/shared/src/hooks';
import {
    checkRequired,
    passwordChecker,
    userNameChecker,
} from '@milesight/shared/src/utils/validators';
import {
    VisibilityIcon,
    VisibilityOffIcon,
    AccountCircleIcon,
    HttpsIcon,
} from '@milesight/shared/src/components';
import { PasswordComplexity } from '../components';

interface UseFormItemsProps {
    mode?: 'login' | 'register';
    /** The latest password */
    latestPassword?: string;
}

export interface FormDataProps {
    username: string;
    password: string;
    /**
     * real-time verification of password complexity
     */
    passwordComplexity?: unknown;
    confirmPassword?: string;
}

const useFormItems = ({ mode = 'login', latestPassword }: UseFormItemsProps) => {
    const { getIntlText } = useI18n();
    const [showPassword, setShowPassword] = useState(false);
    const handleClickShowPassword = useCallback(() => setShowPassword(show => !show), []);

    const formItems = useMemo(() => {
        const props: Partial<TextFieldProps> = {
            fullWidth: true,
            type: 'text',
            size: 'small',
            margin: 'dense',
            sx: { my: 1.5 },
        };
        const registerFields = ['passwordComplexity', 'confirmPassword'];

        const items: ControllerProps<FormDataProps>[] = [
            {
                name: 'username',
                rules: {
                    validate: {
                        checkRequired: checkRequired(),
                        ...userNameChecker(),
                    },
                },
                render({ field: { onChange, value }, fieldState: { error } }) {
                    return (
                        <TextField
                            {...props}
                            placeholder={getIntlText('common.label.username')}
                            error={!!error}
                            helperText={error ? error.message : null}
                            value={value}
                            onChange={onChange}
                            onBlur={event => {
                                const newValue = event?.target?.value;
                                onChange(typeof newValue === 'string' ? newValue.trim() : newValue);
                            }}
                            slotProps={{
                                input: {
                                    startAdornment: (
                                        <InputAdornment position="start">
                                            <AccountCircleIcon />
                                        </InputAdornment>
                                    ),
                                },
                            }}
                        />
                    );
                },
            },
            {
                name: 'password',
                rules: {
                    validate: {
                        checkRequired: checkRequired(),
                        ...passwordChecker(),
                    },
                },
                render({ field: { onChange, value }, fieldState: { error } }) {
                    return (
                        <TextField
                            {...props}
                            autoComplete={mode === 'login' ? undefined : 'new-password'}
                            placeholder={getIntlText('common.label.password')}
                            type={showPassword ? 'text' : 'password'}
                            error={!!error}
                            helperText={error ? error.message : null}
                            value={value}
                            onChange={onChange}
                            slotProps={{
                                input: {
                                    startAdornment: (
                                        <InputAdornment position="start">
                                            <HttpsIcon />
                                        </InputAdornment>
                                    ),
                                    endAdornment: (
                                        <InputAdornment position="end">
                                            <IconButton
                                                aria-label="toggle password visibility"
                                                onClick={handleClickShowPassword}
                                                onMouseDown={(e: any) => e.preventDefault()}
                                                onMouseUp={(e: any) => e.preventDefault()}
                                                edge="end"
                                            >
                                                {showPassword ? (
                                                    <VisibilityOffIcon />
                                                ) : (
                                                    <VisibilityIcon />
                                                )}
                                            </IconButton>
                                        </InputAdornment>
                                    ),
                                },
                            }}
                        />
                    );
                },
            },
            {
                name: 'passwordComplexity',
                render() {
                    return <PasswordComplexity password={latestPassword} />;
                },
            },
            {
                name: 'confirmPassword',
                rules: {
                    validate: {
                        checkRequired: checkRequired(),
                        checkSamePassword(value, formValues) {
                            if (value !== formValues.password) {
                                return getIntlText('valid.input.password.diff');
                            }
                            return true;
                        },
                    },
                },
                render({ field: { onChange, value }, fieldState: { error } }) {
                    return (
                        <TextField
                            {...props}
                            placeholder={getIntlText('common.label.confirm_password')}
                            type={showPassword ? 'text' : 'password'}
                            error={!!error}
                            helperText={error ? error.message : null}
                            value={value}
                            onChange={onChange}
                            slotProps={{
                                input: {
                                    startAdornment: (
                                        <InputAdornment position="start">
                                            <HttpsIcon />
                                        </InputAdornment>
                                    ),
                                    endAdornment: (
                                        <InputAdornment position="end">
                                            <IconButton
                                                aria-label="toggle password visibility"
                                                onClick={handleClickShowPassword}
                                                onMouseDown={(e: any) => e.preventDefault()}
                                                onMouseUp={(e: any) => e.preventDefault()}
                                                edge="end"
                                            >
                                                {showPassword ? (
                                                    <VisibilityOffIcon />
                                                ) : (
                                                    <VisibilityIcon />
                                                )}
                                            </IconButton>
                                        </InputAdornment>
                                    ),
                                },
                            }}
                        />
                    );
                },
            },
        ];

        return items.filter(item => {
            if (registerFields.includes(item.name)) {
                return mode === 'register';
            }

            return true;
        });
    }, [mode, showPassword, getIntlText, handleClickShowPassword, latestPassword]);

    return formItems;
};

export default useFormItems;
