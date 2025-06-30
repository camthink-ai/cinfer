import { useMemo, useRef } from 'react';
import { type ControllerProps } from 'react-hook-form';
import { TextField, FormControl, Autocomplete, Paper, List } from '@mui/material';

import { useI18n } from '@milesight/shared/src/hooks';
import { InfoOutlinedIcon, LoadingWrapper } from '@milesight/shared/src/components';
import { checkRequired, checkIsInt } from '@milesight/shared/src/utils/validators';

import { Tooltip, InputShowCount, SelectVirtualizationList } from '@/components';
import { checkIPWhitelist, ALL_MODELS_SIGN, transformAllModels } from '@/pages/token/utils';
import { type OperateTokenProps } from '../index';
import { useGetModels } from './useGetModels';

import styles from '../style.module.less';

export function useFormItems() {
    const { getIntlText } = useI18n();
    const { modelsLoading, modelOptions } = useGetModels();

    const ipInputRef = useRef<HTMLInputElement | HTMLTextAreaElement>(null);
    const remarkInputRef = useRef<HTMLInputElement | HTMLTextAreaElement>(null);

    const modelsOptionsMock = useMemo(() => {
        return [
            {
                label: getIntlText('token.label.all_models_selection'),
                value: ALL_MODELS_SIGN,
            },
            ...(modelOptions || []),
        ];
    }, [getIntlText, modelOptions]);

    const formItems = useMemo((): ControllerProps<OperateTokenProps>[] => {
        return [
            {
                name: 'name',
                rules: {
                    maxLength: {
                        value: 300,
                        message: getIntlText('valid.input.max_length', {
                            1: 300,
                        }),
                    },
                    validate: {
                        checkRequired: checkRequired(),
                    },
                },
                defaultValue: '',
                render({ field: { onChange, value }, fieldState: { error } }) {
                    return (
                        <TextField
                            required
                            fullWidth
                            type="text"
                            label={getIntlText('common.label.name')}
                            placeholder={getIntlText('common.placeholder.input')}
                            error={!!error}
                            helperText={error ? error.message : null}
                            value={value}
                            onChange={onChange}
                            onBlur={event => {
                                const newValue = event?.target?.value;
                                onChange(typeof newValue === 'string' ? newValue.trim() : newValue);
                            }}
                        />
                    );
                },
            },
            {
                name: 'allowedModels',
                rules: {
                    validate: {
                        checkRequired: checkRequired(),
                    },
                },
                defaultValue: [],
                render({ field: { onChange, value }, fieldState: { error } }) {
                    return (
                        <FormControl fullWidth>
                            <Autocomplete<OptionsProps, true>
                                multiple
                                id="checkboxes-tags-models"
                                size="small"
                                options={modelsOptionsMock}
                                value={modelsOptionsMock.filter(option => {
                                    if (!Array.isArray(value)) {
                                        return false;
                                    }

                                    return value.includes(option.value);
                                })}
                                onChange={(_, selectedOptions) => {
                                    const newValue = (selectedOptions || [])
                                        .map(o => o.value)
                                        .filter(Boolean);

                                    onChange(
                                        transformAllModels(newValue as ApiKey[], value as ApiKey[]),
                                    );
                                }}
                                disableCloseOnSelect
                                disableListWrap
                                renderOption={(props, option, state) => {
                                    Reflect.set(state, 'multiple', true);
                                    return [props, option, state] as React.ReactNode;
                                }}
                                renderInput={params => (
                                    <TextField
                                        {...params}
                                        required
                                        error={!!error}
                                        helperText={error ? error.message : null}
                                        label={getIntlText('token.label.model_permissions')}
                                        placeholder={getIntlText('common.placeholder.select')}
                                    />
                                )}
                                isOptionEqualToValue={(option, valueObj) =>
                                    option.value === valueObj.value
                                }
                                slotProps={{
                                    listbox: {
                                        component: SelectVirtualizationList,
                                    },
                                }}
                                slots={{
                                    paper: modelsLoading
                                        ? () => (
                                              <Paper>
                                                  <LoadingWrapper loading>
                                                      <List
                                                          sx={{
                                                              height: 100,
                                                          }}
                                                      />
                                                  </LoadingWrapper>
                                              </Paper>
                                          )
                                        : undefined,
                                }}
                            />
                        </FormControl>
                    );
                },
            },
            {
                name: 'rateLimit',
                rules: {
                    min: {
                        value: 10,
                        message: getIntlText('valid.input.min_value', {
                            0: 10,
                        }),
                    },
                    max: {
                        value: 100,
                        message: getIntlText('valid.input.max_value', {
                            0: 100,
                        }),
                    },
                    validate: {
                        checkRequired: checkRequired(),
                        checkIsInt: checkIsInt(),
                    },
                },
                defaultValue: 60,
                render({ field: { onChange, value }, fieldState: { error } }) {
                    return (
                        <TextField
                            required
                            fullWidth
                            type="text"
                            label={
                                <div className={styles['label-wrapper']}>
                                    <span>
                                        {getIntlText('token.label.request_frequency_limit')}
                                    </span>
                                    <div className={styles.secondary}>
                                        ({getIntlText('token.label.request_frequency_limit_tip')})
                                    </div>
                                </div>
                            }
                            placeholder={getIntlText('common.placeholder.input')}
                            error={!!error}
                            helperText={error ? error.message : null}
                            value={value}
                            onChange={onChange}
                            onBlur={event => {
                                const newValue = event?.target?.value;
                                onChange(typeof newValue === 'string' ? newValue.trim() : newValue);
                            }}
                        />
                    );
                },
            },
            {
                name: 'monthlyLimit',
                rules: {
                    min: {
                        value: 1,
                        message: getIntlText('valid.input.min_value', {
                            0: 1,
                        }),
                    },
                    max: {
                        value: 999999999999,
                        message: getIntlText('valid.input.max_value', {
                            0: 999999999999,
                        }),
                    },
                    validate: {
                        checkIsInt: checkIsInt(),
                    },
                },
                defaultValue: 1000,
                render({ field: { onChange, value }, fieldState: { error } }) {
                    return (
                        <TextField
                            fullWidth
                            type="text"
                            label={
                                <div className={styles['label-wrapper']}>
                                    <span>{getIntlText('token.label.request_quantity_limit')}</span>
                                    <div className={styles.secondary}>
                                        ({getIntlText('token.label.request_quantity_limit_tip')})
                                    </div>
                                    <div className={styles.tip}>
                                        <Tooltip
                                            title={getIntlText(
                                                'token.label.request_quantity_limit_description',
                                            )}
                                        >
                                            <InfoOutlinedIcon sx={{ width: 16, height: 16 }} />
                                        </Tooltip>
                                    </div>
                                </div>
                            }
                            placeholder={getIntlText('common.placeholder.input')}
                            error={!!error}
                            helperText={error ? error.message : null}
                            value={value}
                            onChange={onChange}
                            onBlur={event => {
                                const newValue = event?.target?.value;
                                onChange(typeof newValue === 'string' ? newValue.trim() : newValue);
                            }}
                        />
                    );
                },
            },
            {
                name: 'ipWhitelist',
                rules: {
                    maxLength: {
                        value: 2000,
                        message: getIntlText('valid.input.max_length', {
                            1: 2000,
                        }),
                    },
                    validate: {
                        checkIPWhitelist: checkIPWhitelist(),
                    },
                },
                defaultValue: '',
                render({ field: { onChange, value }, fieldState: { error } }) {
                    return (
                        <TextField
                            inputRef={ipInputRef}
                            fullWidth
                            type="text"
                            multiline
                            rows={3}
                            label={getIntlText('token.label.ip_whitelist')}
                            placeholder={getIntlText('token.placeholder.ip_whitelist')}
                            error={!!error}
                            helperText={error ? error.message : null}
                            value={value}
                            onChange={onChange}
                            onBlur={event => {
                                const newValue = event?.target?.value;
                                onChange(typeof newValue === 'string' ? newValue.trim() : newValue);
                            }}
                            sx={{
                                '.MuiInputBase-root': {
                                    padding: '0 12px 16px',
                                },
                            }}
                            slotProps={{
                                input: {
                                    endAdornment: (
                                        <InputShowCount
                                            inputRef={ipInputRef}
                                            value={value}
                                            maxLength={2000}
                                        />
                                    ),
                                },
                            }}
                        />
                    );
                },
            },
            {
                name: 'remark',
                rules: {
                    maxLength: {
                        value: 1000,
                        message: getIntlText('valid.input.max_length', {
                            1: 1000,
                        }),
                    },
                },
                defaultValue: '',
                render({ field: { onChange, value }, fieldState: { error } }) {
                    return (
                        <TextField
                            inputRef={remarkInputRef}
                            fullWidth
                            type="text"
                            multiline
                            rows={3}
                            label={getIntlText('common.label.remark')}
                            placeholder={getIntlText('common.placeholder.input')}
                            error={!!error}
                            helperText={error ? error.message : null}
                            value={value}
                            onChange={onChange}
                            onBlur={event => {
                                const newValue = event?.target?.value;
                                onChange(typeof newValue === 'string' ? newValue.trim() : newValue);
                            }}
                            sx={{
                                '.MuiInputBase-root': {
                                    padding: '0 12px 16px',
                                },
                            }}
                            slotProps={{
                                input: {
                                    endAdornment: (
                                        <InputShowCount
                                            inputRef={remarkInputRef}
                                            value={value}
                                            maxLength={1000}
                                        />
                                    ),
                                },
                            }}
                        />
                    );
                },
            },
        ];
    }, [getIntlText, modelsOptionsMock, modelsLoading]);

    return {
        formItems,
    };
}
