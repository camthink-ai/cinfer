import React, { useEffect, useState } from 'react';
import { useForm, Controller, type SubmitHandler } from 'react-hook-form';
import { useMemoizedFn } from 'ahooks';

import classNames from 'classnames';

import { Modal, type ModalProps } from '@milesight/shared/src/components';
import { useI18n } from '@milesight/shared/src/hooks';
import { type FileValueType } from '@/components';
import { useFormItems } from './hooks';

export type OperateModalType = 'add' | 'edit';

export type OperateModelProps = {
    name: string;
    engineType: string;
    modelFile: FileValueType;
    paramsYaml: string;
    remark?: string;
};

interface Props extends Omit<ModalProps, 'onOk'> {
    operateType: OperateModalType;
    /** on form submit */
    onFormSubmit: (data: OperateModelProps, callback: () => void) => Promise<void>;
    data?: OperateModelProps;
}

/**
 * operate model Modal
 */
const OperateModelModal: React.FC<Props> = props => {
    const { visible, onCancel, onFormSubmit, data, operateType, ...restProps } = props;

    const { getIntlText } = useI18n();
    const { control, formState, handleSubmit, reset, setValue } = useForm<OperateModelProps>();

    const [yamlFullscreen, setYamlFullscreen] = useState(false);
    const toggleYamlFullscreen = useMemoizedFn((isFullscreen: boolean) => {
        setYamlFullscreen(Boolean(isFullscreen));
    });

    const { formItems } = useFormItems({
        yamlFullscreen,
        toggleYamlFullscreen,
    });

    const onSubmit: SubmitHandler<OperateModelProps> = async params => {
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
            setValue(k as keyof OperateModelProps, v);
        });
    }, [data, setValue, operateType]);

    return (
        <Modal
            size="lg"
            visible={visible}
            className={classNames({ loading: formState.isSubmitting })}
            onOk={async (...args) => {
                toggleYamlFullscreen(false);
                await handleSubmit(onSubmit)?.(...args);
            }}
            onOkText={getIntlText('common.button.save')}
            onCancel={handleCancel}
            sx={{
                '.MuiDialogContent-root': {
                    position: 'relative',
                },
            }}
            {...restProps}
        >
            {formItems.map(item => (
                <Controller<OperateModelProps> {...item} key={item.name} control={control} />
            ))}
        </Modal>
    );
};

export default OperateModelModal;
