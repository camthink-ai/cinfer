import React, { useEffect } from 'react';
import { useForm, Controller, type SubmitHandler } from 'react-hook-form';
import { useMemoizedFn } from 'ahooks';

import classNames from 'classnames';

import { Modal, type ModalProps } from '@milesight/shared/src/components';
import { useI18n } from '@milesight/shared/src/hooks';
import { useFormItems } from './hooks';

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
 * operate token Modal
 */
const OperateTokenModal: React.FC<Props> = props => {
    const { visible, onCancel, onFormSubmit, data, operateType, ...restProps } = props;

    const { getIntlText } = useI18n();
    const { control, formState, handleSubmit, reset, setValue } = useForm<OperateTokenProps>();
    const { formItems } = useFormItems();

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

export default OperateTokenModal;
