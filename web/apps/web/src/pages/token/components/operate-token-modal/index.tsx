import React, { useMemo, useEffect } from 'react';
import { useForm, Controller, type SubmitHandler, type ControllerProps } from 'react-hook-form';
import { useMemoizedFn } from 'ahooks';
import {
    TextField,
    MenuItem,
    Checkbox,
    ListItemText,
    FormControl,
    Select as MuiSelect,
    InputLabel,
    FormHelperText,
} from '@mui/material';
import classNames from 'classnames';
import { isEmpty } from 'lodash-es';

import { Modal, type ModalProps, InfoOutlinedIcon } from '@milesight/shared/src/components';
import { checkRequired, checkIsInt } from '@milesight/shared/src/utils/validators';
import { useI18n } from '@milesight/shared/src/hooks';

import { Tooltip } from '@/components';

import styles from './style.module.less';

export type OperateModalType = 'add' | 'edit';

export type OperateTokenProps = {
    name: string;
    allowedModels: ApiKey[];
    rateLimit: string;
    monthlyLimit?: string;
    ipWhitelist?: string;
    remark?: string;
};

interface Props extends Omit<ModalProps, 'onOk'> {
    operateType: OperateModalType;
    /** on form submit */
    onFormSubmit: (data: OperateTokenProps, callback: () => void) => Promise<void>;
    data?: OperateTokenProps;
}

/**
 * operate user Modal
 */
const OperateUserModal: React.FC<Props> = props => {
    const { visible, onCancel, onFormSubmit, data, operateType, ...restProps } = props;

    const { getIntlText } = useI18n();
    const { control, formState, handleSubmit, reset, setValue } = useForm<OperateTokenProps>();

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
                        <FormControl fullWidth error={!!error}>
                            <InputLabel id="select-label" size="small" required error={!!error}>
                                {getIntlText('token.label.model_permissions')}
                            </InputLabel>
                            <MuiSelect
                                labelId="multiple-checkbox-label"
                                id="multiple-checkbox"
                                multiple
                                value={value}
                                onChange={onChange}
                                displayEmpty
                                renderValue={value => {
                                    if (!Array.isArray(value) || isEmpty(value)) {
                                        return (
                                            <span className={styles['select-placeholder']}>
                                                {getIntlText('common.placeholder.select')}
                                            </span>
                                        );
                                    }

                                    return ((value as ApiKey[]) || []).join(',');
                                }}
                            >
                                {['ALL', 'model1', 'model2'].map(name => (
                                    <MenuItem key={name} value={name}>
                                        <Checkbox
                                            checked={((value as ApiKey[]) || []).includes(name)}
                                        />
                                        <ListItemText primary={name} />
                                    </MenuItem>
                                ))}
                            </MuiSelect>
                            {!!error && <FormHelperText>{error.message}</FormHelperText>}
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
                    validate: {
                        checkIsInt: checkIsInt(),
                    },
                },
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
                },
                defaultValue: '',
                render({ field: { onChange, value }, fieldState: { error } }) {
                    return (
                        <TextField
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
                                    padding: '0 12px 12px',
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
                                    padding: '0 12px 12px',
                                },
                            }}
                        />
                    );
                },
            },
        ];
    }, [getIntlText]);

    const onSubmit: SubmitHandler<OperateTokenProps> = async params => {
        await onFormSubmit(params, () => {
            reset();
        });
    };

    const handleCancel = useMemoizedFn(() => {
        reset();
        onCancel?.();
    });

    /**
     * initial form value
     */
    useEffect(() => {
        if (operateType !== 'edit') {
            return;
        }

        Object.entries(data || {}).forEach(([k, v]) => {
            setValue(k as keyof OperateTokenProps, v);
        });
    }, [data, setValue, operateType]);

    return (
        <Modal
            size="lg"
            visible={visible}
            className={classNames({ loading: formState.isSubmitting })}
            onOk={handleSubmit(onSubmit)}
            onOkText={getIntlText('common.button.save')}
            onCancel={handleCancel}
            {...restProps}
        >
            {formItems.map(item => (
                <Controller<OperateTokenProps> {...item} key={item.name} control={control} />
            ))}
        </Modal>
    );
};

export default OperateUserModal;
