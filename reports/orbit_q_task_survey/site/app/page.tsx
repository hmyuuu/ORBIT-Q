import { SurveyAtlas } from "./SurveyAtlas";
import { surveyData } from "../lib/survey";

export default function Home() {
  return <SurveyAtlas data={surveyData} />;
}
