import { useState } from 'react';
import { useMemoizedFn } from 'ahooks';

import { useI18n } from '@milesight/shared/src/hooks';
import { toast } from '@milesight/shared/src/components';

import { tokenAPI, awaitWrap, isRequestSuccess, type TokenItemProps } from '@/services/http';
import { type OperateTokenProps, type OperateModalType } from '../components/operate-token-modal';
import { convertData } from '../utils';

/**
 * use token modal data resolved
 */
export default function useTokenModal(getAllTokens?: () => void) {
    const { getIntlText } = useI18n();

    const [tokenModalVisible, setTokenModalVisible] = useState(false);
    const [operateType, setOperateType] = useState<OperateModalType>('add');
    const [modalTitle, setModalTitle] = useState(getIntlText('token.title.modal_add_token'));
    const [currentToken, setCurrentToken] = useState<TokenItemProps>();

    const hideModal = useMemoizedFn(() => {
        setTokenModalVisible(false);
    });

    const openAddToken = useMemoizedFn(() => {
        setOperateType('add');
        setModalTitle(getIntlText('token.title.modal_add_token'));
        setTokenModalVisible(true);
    });

    const openEditToken = useMemoizedFn((item: TokenItemProps) => {
        setOperateType('edit');
        setModalTitle(getIntlText('token.title.modal_edit_token'));
        setTokenModalVisible(true);
        setCurrentToken(item);
    });

    const handleAddToken = useMemoizedFn(async (data: OperateTokenProps, callback: () => void) => {
        const [error, resp] = await awaitWrap(tokenAPI.addNewToken(convertData(data)));

        if (error || !isRequestSuccess(resp)) {
            return;
        }

        getAllTokens?.();
        setTokenModalVisible(false);
        toast.success(getIntlText('common.message.add_success'));
        callback?.();
    });

    const handleEditToken = useMemoizedFn(async (data: OperateTokenProps, callback: () => void) => {
        if (!currentToken) return;

        const [error, resp] = await awaitWrap(
            tokenAPI.updateToken({
                access_token_id: currentToken?.id,
                ...convertData(data),
            }),
        );

        if (error || !isRequestSuccess(resp)) {
            return;
        }

        getAllTokens?.();
        setTokenModalVisible(false);
        toast.success(getIntlText('common.message.operation_success'));
        callback?.();
    });

    const onFormSubmit = useMemoizedFn(async (data: OperateTokenProps, callback: () => void) => {
        if (!data) return;

        if (operateType === 'add') {
            await handleAddToken(data, callback);
            return;
        }

        await handleEditToken(data, callback);
    });

    return {
        tokenModalVisible,
        openAddToken,
        openEditToken,
        hideModal,
        operateType,
        setOperateType,
        onFormSubmit,
        modalTitle,
        currentToken,
    };
}
