import { useRequest, useMemoizedFn } from 'ahooks';

import { useI18n } from '@milesight/shared/src/hooks';
import { toast } from '@milesight/shared/src/components';

import {
    modelAPI,
    isRequestSuccess,
    awaitWrap,
    getResponseData,
    type ModelItemProps,
} from '@/services/http';
import { useGlobalStore } from '@/stores';
import { useConfirm } from '@/components';

export default function useModel(getAllModels?: () => void) {
    const { setInferEngines } = useGlobalStore();
    const confirm = useConfirm();
    const { getIntlText } = useI18n();

    const { run: getInferEngines } = useRequest(
        async () => {
            const [err, resp] = await awaitWrap(modelAPI.getEngines());
            if (err || !isRequestSuccess(resp)) {
                return;
            }

            const data = getResponseData(resp);
            setInferEngines(data);
        },
        {
            manual: true,
        },
    );

    const handleDeleteModel = useMemoizedFn((record: ModelItemProps) => {
        confirm({
            title: getIntlText('common.label.delete'),
            description: getIntlText('model.tip.delete_model'),
            confirmButtonText: getIntlText('common.label.delete'),
            confirmButtonProps: {
                color: 'error',
            },
            onConfirm: async () => {
                if (!record?.id) return;

                const [err, resp] = await awaitWrap(
                    modelAPI.deleteModel({
                        model_id: record.id,
                    }),
                );

                if (err || !isRequestSuccess(resp)) {
                    return;
                }

                getAllModels?.();
                toast.success(getIntlText('common.message.delete_success'));
            },
        });
    });

    const handlePublishModel = useMemoizedFn((record: ModelItemProps, isPublish?: boolean) => {
        const title = isPublish
            ? getIntlText('common.label.publish')
            : getIntlText('common.label.unpublish');

        const description = isPublish
            ? getIntlText('model.tip.to_publish')
            : getIntlText('model.tip.to_unpublish');

        confirm({
            title,
            description,
            confirmButtonText: getIntlText('common.button.confirm'),
            onConfirm: async () => {
                if (!record?.id) return;

                const whetherToPublish = isPublish
                    ? modelAPI.publishModel
                    : modelAPI.unpublishModel;
                const [err, resp] = await awaitWrap(
                    whetherToPublish({
                        model_id: record.id,
                    }),
                );

                if (err || !isRequestSuccess(resp)) {
                    return;
                }

                getAllModels?.();
                toast.success(getIntlText('common.message.operation_success'));
            },
        });
    });

    return {
        getInferEngines,
        /** To delete model */
        handleDeleteModel,
        /** Weather to publish */
        handlePublishModel,
    };
}
