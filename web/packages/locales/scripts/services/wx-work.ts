import axios from 'axios';
import { PHRASE_WX_WORK_KEY, PHRASE_WX_WORK_MENTIONED_MOBILES } from '../config';

const url = `https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=${PHRASE_WX_WORK_KEY}`;

export const sendMessage = async (options: { content: string; mentionedMobiles?: string[] }) => {
    const { content, mentionedMobiles = PHRASE_WX_WORK_MENTIONED_MOBILES?.split(',') } = options;

    return axios({
        url,
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        data: {
            msgtype: 'text',
            text: {
                content,
                mentioned_mobile_list: mentionedMobiles,
            },
        },
    });
};
