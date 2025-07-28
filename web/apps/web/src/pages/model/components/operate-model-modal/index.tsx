import React, { useState } from 'react';
import { useForm, Controller, type SubmitHandler } from 'react-hook-form';
import { useMemoizedFn, useRequest } from 'ahooks';

import classNames from 'classnames';

import { Modal, type ModalProps, LoadingWrapper } from '@milesight/shared/src/components';
import { useI18n } from '@milesight/shared/src/hooks';
import { type FileValueType } from '@/components';
import { modelAPI, getResponseData, isRequestSuccess, awaitWrap } from '@/services/http';
import { useFormItems } from './hooks';
import { convertDataToDisplay } from '../../utils';

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
    id?: ApiKey;
}

/**
 * operate model Modal
 */
const OperateModelModal: React.FC<Props> = props => {
    const { visible, onCancel, onFormSubmit, id, operateType, ...restProps } = props;

    const { getIntlText } = useI18n();
    const { control, formState, handleSubmit, reset, setValue, watch } =
        useForm<OperateModelProps>();

    const [yamlFullscreen, setYamlFullscreen] = useState(false);
    const toggleYamlFullscreen = useMemoizedFn((isFullscreen: boolean) => {
        setYamlFullscreen(Boolean(isFullscreen));
    });

    const { formItems } = useFormItems({
        yamlFullscreen,
        toggleYamlFullscreen,
        engineType: watch('engineType'),
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
    const { loading } = useRequest(
        async () => {
            if (operateType !== 'edit' || !id) {
                return;
            }

            const [error, resp] = await awaitWrap(
                modelAPI.getModelDetail({
                    model_id: id,
                }),
            );
            if (error || !isRequestSuccess(resp)) {
                return;
            }

            const data = getResponseData(resp);
            const newData = convertDataToDisplay(data);

            Object.entries(newData).forEach(([k, v]) => {
                setValue(k as keyof OperateModelProps, v);
            });
        },
        {
            refreshDeps: [id, operateType],
        },
    );

    const renderContent = () => {
        const coreContent = formItems.map(item => (
            <Controller<OperateModelProps> {...item} key={item.name} control={control} />
        ));

        if (yamlFullscreen) {
            return coreContent;
        }

        return <LoadingWrapper loading={loading}>{coreContent}</LoadingWrapper>;
    };

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
                '.MuiDialog-paper': {
                    height: yamlFullscreen ? 'calc(100% - 96px)' : undefined,
                },
            }}
            okButtonProps={{
                disabled: loading,
            }}
            {...restProps}
        >
            {renderContent()}
        </Modal>
    );
};

export default OperateModelModal;
