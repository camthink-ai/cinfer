import { useState } from 'react';
import { useMemoizedFn } from 'ahooks';

import { useI18n, useCopy } from '@milesight/shared/src/hooks';
import { toast, CheckCircleIcon } from '@milesight/shared/src/components';

import { useConfirm } from '@/components';
import {
    tokenAPI,
    awaitWrap,
    isRequestSuccess,
    getResponseData,
    type TokenItemProps,
} from '@/services/http';
import { type OperateTokenProps, type OperateModalType } from '../components/operate-token-modal';
import { convertData } from '../utils';

import styles from '../style.module.less';

/**
 * use token modal data resolved
 */
export default function useTokenModal(getAllTokens?: () => void) {
    const { getIntlText } = useI18n();
    const confirm = useConfirm();
    const { handleCopy } = useCopy();

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
        callback?.();

        const newToken = getResponseData(resp);
        if (!newToken) return;

        confirm({
            title: getIntlText('token.title.token_add_successful'),
            description: (
                <div className={styles['add-success-modal']}>
                    <div className={styles.tip}>{getIntlText('token.title.token_save_tip')}</div>
                    <div className={styles.title}>{getIntlText('token.title.token_value')}</div>
                    <div id="token-value__cam_think_web" className={styles['token-value']}>
                        {newToken.token}
                    </div>
                </div>
            ),
            icon: <CheckCircleIcon color="success" />,
            confirmButtonText: getIntlText('common.label.copy'),
            cancelButtonText: getIntlText('common.label.close'),
            onConfirm: async () => {
                if (navigator?.clipboard) {
                    handleCopy(newToken.token);
                    return;
                }

                const div = document.getElementById('token-value__cam_think_web');
                if (window.getSelection && div) {
                    const selection = window.getSelection();
                    const range = document.createRange();
                    range.selectNodeContents(div);
                    selection?.removeAllRanges();
                    selection?.addRange(range);
                }

                try {
                    document.execCommand('copy');
                } catch (err) {
                    // eslint-disable-next-line no-console
                    console.warn(err);
                }
            },
        });
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
