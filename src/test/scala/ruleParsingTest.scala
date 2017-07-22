import scala.collection.mutable.{ListBuffer, MutableList, Map => mutableMap}

object ruleParsingTest {

	def main(args: Array[String]): Unit = {
		val rule = "t_stress(P,Yes) :- assign(Yes,yes),person(P) {r1}."

		/*
			Convert a rule into a list of string of variables and factors
			Output data:
				- List(String)
				- List(Tuple(String, String, String))
		 */

		var variables = MutableList[String]()
		var factors = mutableMap[String, (String, String)]()

		var tokens = rule.split(":-")
		var head = tokens(0)
		//head = t_stress(P,Yes)
		var body = tokens(1)
		//body =  assign(Yes,yes),person(P) {r1}.

		//Fix the head
		var headFactor = head.split("\\(")(0)  //DONE!
		//headFactor = t_stress (DONE)
		var headRest = head.slice(headFactor.length,head.length)
		//headRest = (P,Yes)
		headRest = headRest.replace("(","").replace(")","")
		//headRest = P,Yes
		var headVars = headRest.split(",")    //DONE!
		for (headVar <- headVars) {
			if (!variables.contains(headVar)) variables += headVar.trim
		}

		//headVars.mkString(" and ") = P and Yes

		//Fix the body
		body = body.slice(0,body.length - 6).trim
		//body = assign(Yes,yes),person(P)
		var bodyTokens = body.split("\\),")
		//bodyTokens.mkString("|") = assign(Yes,yes|person(P)

		for (token <- bodyTokens) {

			//Get the tokenhead
			var tokenHead = token.split("\\(")(0).trim
			//tokenHead = person

			//Get the body
			var tokenBody = token.split("\\(")(1).trim
			//tokenBody = Yes,yes || P)
			if (tokenBody(tokenBody.length-1) == ')') {
				//If here, remove the trailing bracket
				tokenBody = tokenBody.slice(0,tokenBody.length-1)
			}
			//tokenBody = Yes,yes || P


			if (tokenBody.contains(',')) {
				//Binary token

				//Find the left and right variable.
				//Append both of them to a mutable map.

				var leftVar = tokenBody.split(',')(0).trim
				var rightVar = tokenBody.split(',')(1).trim

				if (!variables.contains(leftVar)) variables += leftVar
				if (!variables.contains(rightVar)) variables += rightVar

				factors += (tokenHead -> (leftVar, rightVar))
			}
			else {
				//Unary token
				var leftVar = tokenBody.trim
				if (!variables.contains(leftVar)) variables += leftVar
				factors += (tokenHead -> (leftVar, null))
			}

		}

		(variables, factors)


	}
}
