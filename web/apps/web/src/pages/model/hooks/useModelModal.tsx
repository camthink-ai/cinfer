import { useState } from 'react';
import { useMemoizedFn } from 'ahooks';

import { useI18n } from '@milesight/shared/src/hooks';
import { toast } from '@milesight/shared/src/components';

import { modelAPI, awaitWrap, isRequestSuccess, type ModelItemProps } from '@/services/http';
import { convertAddModelData, convertEditModelData } from '../utils';
import { type OperateModelProps, type OperateModalType } from '../components/operate-model-modal';

/**
 * use token modal data resolved
 */
export default function useModelModal(getAllModels?: () => void, getInferEngines?: () => void) {
    const { getIntlText } = useI18n();

    const [modelModalVisible, setModelModalVisible] = useState(false);
    const [operateType, setOperateType] = useState<OperateModalType>('add');
    const [modalTitle, setModalTitle] = useState(getIntlText('model.title.modal_add_model'));
    const [currentModel, setCurrentModel] = useState<ModelItemProps>();

    const hideModal = useMemoizedFn(() => {
        setModelModalVisible(false);
    });

    const openAddModel = useMemoizedFn(() => {
        setOperateType('add');
        setModalTitle(getIntlText('model.title.modal_add_model'));
        setModelModalVisible(true);
        getInferEngines?.();
    });

    const openEditModel = useMemoizedFn((item: ModelItemProps) => {
        setOperateType('edit');
        setModalTitle(getIntlText('model.title.modal_edit_model'));
        setModelModalVisible(true);
        setCurrentModel(item);
        getInferEngines?.();
    });

    const handleAddModel = useMemoizedFn(async (data: OperateModelProps, callback: () => void) => {
        const newData = convertAddModelData(data);
        if (!newData) return;

        const [error, resp] = await awaitWrap(modelAPI.addModel(newData));

        if (error || !isRequestSuccess(resp)) {
            return;
        }

        getAllModels?.();
        setModelModalVisible(false);
        toast.success(getIntlText('common.message.add_successful'));
        callback?.();
    });

    const handleEditModel = useMemoizedFn(async (data: OperateModelProps, callback: () => void) => {
        if (!currentModel?.id) return;

        const [error, resp] = await awaitWrap(
            modelAPI.updateModel({
                model_id: currentModel.id,
                ...convertEditModelData(data),
            }),
        );

        if (error || !isRequestSuccess(resp)) {
            return;
        }

        getAllModels?.();
        setModelModalVisible(false);
        toast.success(getIntlText('common.message.operation_success'));
        callback?.();
    });

    const onFormSubmit = useMemoizedFn(async (data: OperateModelProps, callback: () => void) => {
        if (!data) return;

        if (operateType === 'add') {
            await handleAddModel(data, callback);
            return;
        }

        await handleEditModel(data, callback);
    });

    return {
        modelModalVisible,
        openAddModel,
        openEditModel,
        hideModal,
        operateType,
        setOperateType,
        onFormSubmit,
        modalTitle,
        currentModel,
    };
}
